"""Tests for ESMFold2Backend's model_opt path (the ESMFold2 inference optimization kit).

Fully mocked — no torch, esm or esmfold2_opt required.  The GPU check that
``exact`` is byte-identical to ``off`` lives in
``tests/test_gpu_model_opt_esmfold2.py`` (gated on SHEAF_GPU_KIT_TEST=1).

Covers:
  - model_opt=None never touches the kit and keeps a builder per request
  - mode="off" selects the fused backend with chunking off, kit never imported
  - kit modes call esmfold2_opt.enable(mode, variant=..., strict=True) before
    the model is built, apply the kit to the loaded model with the builder
    that serves requests, then warm up once and check status()
  - the kit's lines are captured and re-logged
  - a refusal (enable, apply_to's exit, partial levers, kit missing) raises
    ModelOptNotActiveError at load(), not at request time
  - model names map to kit variants; unknown names are rejected
  - one mode and one variant per process
"""

from __future__ import annotations

import logging
import os
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.structure import ChainInput, StructureRequest
from sheaf.model_opt import (
    ModelOptConfig,
    ModelOptNotActiveError,
    _reset_claims,
    apply_model_opt,
)


class _Result:
    def __init__(self) -> None:
        self.complex = MagicMock()
        self.complex.to_mmcif.return_value = "CIF"
        self.plddt = [0.9, 0.8]
        self.ptm = 0.7
        self.iptm = None
        self.pae = None
        self.sample_scores = None


class _Builder:
    instances: list[_Builder] = []

    def __init__(self) -> None:
        self.folds: list[dict] = []
        _Builder.instances.append(self)

    def fold(self, model: Any, spi: Any, **kw: Any) -> _Result:
        self.folds.append(kw)
        return _Result()


def _make_upstream(events: list[str]) -> tuple[dict[str, Any], MagicMock]:
    model = MagicMock()
    model.to.return_value = model

    def _set_backend(name: str) -> None:
        events.append(f"set_kernel_backend:{name}")

    def _set_chunk(size: Any) -> None:
        events.append(f"set_chunk_size:{size}")

    model.set_kernel_backend.side_effect = _set_backend
    model.set_chunk_size.side_effect = _set_chunk

    ESMFold2Model = MagicMock()

    def _from_pretrained(name: str) -> Any:
        events.append(f"from_pretrained:{name}")
        return model

    ESMFold2Model.from_pretrained.side_effect = _from_pretrained

    esmfold2 = ModuleType("esm.models.esmfold2")
    esmfold2.ESMFold2InputBuilder = _Builder  # type: ignore[attr-defined]
    esmfold2.ProteinInput = MagicMock()  # type: ignore[attr-defined]
    esmfold2.StructurePredictionInput = MagicMock()  # type: ignore[attr-defined]
    modeling = ModuleType("transformers.models.esmfold2.modeling_esmfold2")
    modeling.ESMFold2Model = ESMFold2Model  # type: ignore[attr-defined]
    mods = {
        "esm": ModuleType("esm"),
        "esm.models": ModuleType("esm.models"),
        "esm.models.esmfold2": esmfold2,
        "transformers": ModuleType("transformers"),
        "transformers.models": ModuleType("transformers.models"),
        "transformers.models.esmfold2": ModuleType("transformers.models.esmfold2"),
        "transformers.models.esmfold2.modeling_esmfold2": modeling,
    }
    return mods, model


def _make_torch() -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = "2.13.0+cu130"  # type: ignore[attr-defined]
    torch.version = MagicMock(cuda="13.0")  # type: ignore[attr-defined]
    torch.cuda = MagicMock()  # type: ignore[attr-defined]
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_capability.return_value = (9, 0)
    return torch


class _FakeKit:
    """esmfold2_opt stand-in recording calls and printing the kit's lines."""

    def __init__(
        self,
        events: list[str],
        refuse: str | None = None,
        partial: list[str] | None = None,
        exit_on_apply: bool = False,
    ) -> None:
        self.events = events
        self.refuse = refuse
        self.partial = partial or []
        self.exit_on_apply = exit_on_apply
        self.applied_with: tuple[Any, Any] | None = None
        self.mod = ModuleType("esmfold2_opt")

        class ActivationError(RuntimeError):
            pass

        self.mod.ActivationError = ActivationError  # type: ignore[attr-defined]
        self.mod.enable = self.enable  # type: ignore[attr-defined]
        self.mod.status = self.status  # type: ignore[attr-defined]
        self.mod.stack = MagicMock()  # type: ignore[attr-defined]
        self.mod.stack.apply_to.side_effect = self.apply_to  # type: ignore[attr-defined]

    def enable(self, mode: str, **kw: Any) -> dict:
        self.events.append(f"enable:{mode}:{kw}")
        if self.refuse:
            print(f"[esmfold2-opt] NOT ACTIVE: {self.refuse}")
            raise self.mod.ActivationError(self.refuse)  # type: ignore[attr-defined]
        print(f"[esmfold2-opt] ACTIVE mode={mode} variant={kw.get('variant')}")
        return {"active": True, "mode": mode}

    def apply_to(self, model: Any, builder: Any) -> dict:
        self.events.append("apply_to")
        self.applied_with = (model, builder)
        if self.exit_on_apply:
            print("[esmfold2-opt] NOT ACTIVE: flash-attention path not live")
            raise SystemExit(3)
        print("[esmfold2-opt] APPLIED model#1 levers_applied=fused,tg")
        return {}

    def status(self) -> dict:
        self.events.append("status")
        return {
            "active": True,
            "mode": "exact",
            "variant": "full_nomsa",
            "levers_applied": ["fused", "tg"],
            "partial": self.partial,
        }


@pytest.fixture(autouse=True)
def _clean():
    _reset_claims()
    _Builder.instances.clear()
    yield
    _reset_claims()


def _backend(mode: str | None, model_name: str = "biohub/ESMFold2", **cfg: Any):
    from sheaf.backends.esmfold2 import ESMFold2Backend

    backend = ESMFold2Backend(model_name=model_name, device="cuda")
    if mode is not None:
        apply_model_opt(backend, ModelOptConfig(mode=mode, **cfg), "dep")  # type: ignore[arg-type]
    return backend


def _load(backend: Any, events: list[str], kit: _FakeKit | None = None) -> MagicMock:
    mods, model = _make_upstream(events)
    mods["torch"] = _make_torch()
    if kit is not None:
        mods["esmfold2_opt"] = kit.mod
    with patch.dict(sys.modules, mods):
        backend.load()
    return model


def _request(**kw: Any) -> StructureRequest:
    return StructureRequest(
        model_name="esmfold2", chains=[ChainInput(chain_id="A", sequence="MKTII")], **kw
    )


# ---------------------------------------------------------------------------
# Mode support + legacy path
# ---------------------------------------------------------------------------


def test_supported_modes_are_off_exact_fast() -> None:
    assert _backend(None).supported_opt_modes() == frozenset({"off", "exact", "fast"})


def test_big_is_rejected_at_apply() -> None:
    with pytest.raises(ValueError, match="'big'"):
        _backend("big")


def test_model_opt_none_keeps_legacy_path() -> None:
    events: list[str] = []
    kit = _FakeKit(events)
    backend = _backend(None)
    _load(backend, events, kit)
    assert events == ["from_pretrained:biohub/ESMFold2"]  # no backend calls, no kit
    backend.predict(_request())
    backend.predict(_request())
    assert len(_Builder.instances) == 2  # a builder per request, as before


# ---------------------------------------------------------------------------
# mode="off"
# ---------------------------------------------------------------------------


def test_off_selects_fused_backend_without_kit() -> None:
    events: list[str] = []
    backend = _backend("off")
    _load(backend, events, _FakeKit(events))
    assert events == [
        "from_pretrained:biohub/ESMFold2",
        "set_kernel_backend:fused",
        "set_chunk_size:None",
    ]
    assert backend.model_opt_report is None
    assert _Builder.instances[0].folds == []  # no warm-up under off


# ---------------------------------------------------------------------------
# Kit modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_kit_mode_enables_before_build_applies_and_warms_up(mode: str, caplog) -> None:
    caplog.set_level(logging.INFO, logger="sheaf.model_opt")
    events: list[str] = []
    kit = _FakeKit(events)
    backend = _backend(mode)
    model = _load(backend, events, kit)
    assert events == [
        f"enable:{mode}:{{'variant': 'full_nomsa', 'strict': True}}",
        "from_pretrained:biohub/ESMFold2",
        "apply_to",
        "status",
    ]
    (builder,) = _Builder.instances
    assert kit.applied_with == (model, builder)
    assert len(builder.folds) == 1  # the warm-up fold, before status()
    assert backend.model_opt_report is not None and backend.model_opt_report["active"]
    logged = [r.getMessage() for r in caplog.records if r.name == "sheaf.model_opt"]
    assert any("ACTIVE mode=" in m for m in logged)
    assert any("APPLIED" in m for m in logged)


def test_kit_mode_reuses_the_applied_builder_for_requests() -> None:
    events: list[str] = []
    backend = _backend("exact")
    _load(backend, events, _FakeKit(events))
    backend.predict(_request(seed=3))
    (builder,) = _Builder.instances
    assert len(builder.folds) == 2 and builder.folds[-1]["seed"] == 3


def test_refusal_at_enable_raises_at_load() -> None:
    events: list[str] = []
    backend = _backend("exact")
    with pytest.raises(ModelOptNotActiveError, match="no visible GPU"):
        _load(backend, events, _FakeKit(events, refuse="no visible GPU"))
    assert "from_pretrained:biohub/ESMFold2" not in events


def test_apply_to_exit_becomes_not_active() -> None:
    events: list[str] = []
    backend = _backend("fast")
    with pytest.raises(ModelOptNotActiveError, match="exit 3"):
        _load(backend, events, _FakeKit(events, exit_on_apply=True))


def test_partial_levers_after_warmup_raise() -> None:
    events: list[str] = []
    backend = _backend("exact")
    with pytest.raises(ModelOptNotActiveError, match="partial=\\['tg'\\]"):
        _load(backend, events, _FakeKit(events, partial=["tg"]))


def test_kit_not_installed_raises_not_active() -> None:
    events: list[str] = []
    backend = _backend("exact")
    mods, _ = _make_upstream(events)
    mods["torch"] = _make_torch()
    with patch.dict(sys.modules, mods), patch.dict(sys.modules, {"esmfold2_opt": None}):
        with pytest.raises(ModelOptNotActiveError, match="not installed"):
            backend.load()


def test_levers_off_are_exported(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    events: list[str] = []
    backend = _backend("fast", levers_off=["tx", "dit"])
    _load(backend, events, _FakeKit(events))
    assert os.environ["MODEL_OPT_LEVERS_OFF"] == "tx,dit"


# ---------------------------------------------------------------------------
# Names, devices, process isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("biohub/ESMFold2", "full_nomsa"),
        ("Biohub/esmfold2", "full_nomsa"),
        ("biohub/ESMFold2-Fast", "fast"),
    ],
)
def test_model_name_maps_to_kit_variant(name: str, variant: str) -> None:
    events: list[str] = []
    _load(_backend("exact", model_name=name), events, _FakeKit(events))
    assert f"'variant': '{variant}'" in events[0]


def test_unknown_model_name_rejected() -> None:
    with pytest.raises(ValueError, match="biohub/ESMFold2"):
        _load(_backend("exact", model_name="someone/else"), [])


def test_kit_mode_on_cpu_rejected() -> None:
    from sheaf.backends.esmfold2 import ESMFold2Backend

    backend = ESMFold2Backend(device="cpu")
    apply_model_opt(backend, ModelOptConfig(mode="fast"), "dep")
    with pytest.raises(ValueError, match="CUDA"):
        _load(backend, [])


def test_second_mode_in_same_process_refused() -> None:
    events: list[str] = []
    _load(_backend("exact"), events, _FakeKit(events))
    with pytest.raises(ModelOptNotActiveError, match="process-wide"):
        _load(_backend("fast"), events, _FakeKit(events))


def test_second_variant_in_same_process_refused() -> None:
    events: list[str] = []
    _load(_backend("exact"), events, _FakeKit(events))
    with pytest.raises(ModelOptNotActiveError, match="process-wide"):
        fast_variant = _backend("exact", model_name="biohub/ESMFold2-Fast")
        _load(fast_variant, events, _FakeKit(events))
