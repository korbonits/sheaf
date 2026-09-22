"""Tests for ESMCBackend's model_opt path (the ESM C inference optimization kit).

Fully mocked — no torch, esm or esmc_opt required.  The GPU check that
``exact`` is byte-identical to ``off`` lives in
``tests/test_gpu_model_opt_esmc.py`` (gated on SHEAF_GPU_KIT_TEST=1).

Covers:
  - model_opt=None never touches the esm SDK or the kit (legacy path)
  - mode="off" loads through esm's ESMC.from_pretrained, kit never imported
  - mode="exact" calls esmc_opt.enable(..., regime="batched", strict=True)
    before the client is built, then warms up once at load
  - the kit's ACTIVE / APPLIED lines are captured and re-logged
  - a NOT ACTIVE refusal (enable, from_pretrained, or partial levers after the
    warm-up) raises ModelOptNotActiveError at load(), not at request time
  - the SDK path runs input_ids + attention_mask only, under bf16 autocast
  - model names map to SDK names / kit variants; unknown names are rejected
  - a second, different kit mode in the same process is refused
  - device_map and levers_off are rejected on this path
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.protein_language import ProteinLanguageRequest
from sheaf.model_opt import (
    ModelOptConfig,
    ModelOptNotActiveError,
    _reset_claims,
    apply_model_opt,
)

_HIDDEN = 16
_VOCAB = 33


class FakeTensor:
    def __init__(self, data: Any, dtype: Any = np.float32) -> None:
        self._data = np.asarray(data, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(self._data[key])  # type: ignore[index]

    def __iter__(self):  # stacked hidden_states: iterate over layers
        return (FakeTensor(x) for x in self._data)

    def sum(self, dim: int) -> FakeTensor:
        return FakeTensor(self._data.sum(axis=dim))

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return FakeTensor(self._data.astype(np.float32))

    def int(self) -> FakeTensor:
        return FakeTensor(self._data.astype(np.int64), dtype=np.int64)

    def tolist(self) -> list:
        return self._data.tolist()

    def to(self, _device: Any) -> FakeTensor:
        return self


class _Ctx:
    entered: list[tuple] = []

    def __init__(self, *args: Any, **kw: Any) -> None:
        self.args = (args, kw)

    def __enter__(self) -> _Ctx:
        _Ctx.entered.append(self.args)
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _make_torch() -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = "2.11.0+cu130"  # type: ignore[attr-defined]
    torch.inference_mode = _Ctx  # type: ignore[attr-defined]
    torch.autocast = _Ctx  # type: ignore[attr-defined]
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.device = lambda d: f"device({d})"  # type: ignore[attr-defined]
    torch.version = MagicMock(cuda="13.0")  # type: ignore[attr-defined]
    torch.cuda = MagicMock()  # type: ignore[attr-defined]
    torch.cuda.is_available.return_value = True
    torch.cuda.get_device_capability.return_value = (9, 0)
    return torch


class _Tokenized(dict):
    """BatchEncoding-like: the SDK tokenizer also returns extra keys."""


def _make_client(real_lens: tuple[int, ...] = (6, 5), padded: int = 8) -> Any:
    n = len(real_lens)
    mask = np.zeros((n, padded), dtype=np.int64)
    for i, k in enumerate(real_lens):
        mask[i, :k] = 1

    tokenizer = MagicMock()
    tokenizer.return_value = _Tokenized(
        input_ids=FakeTensor(np.zeros((n, padded)), dtype=np.int64),
        attention_mask=FakeTensor(mask, dtype=np.int64),
        token_type_ids=FakeTensor(np.zeros((n, padded)), dtype=np.int64),
    )

    out = MagicMock(spec=["logits", "hidden_states"])
    out.logits = FakeTensor(np.full((n, padded, _VOCAB), 0.25))
    # esm 3.4.0 native EsmcMaskedLMOutput: hidden_states is ONE stacked tensor
    # (n_layers + 1, batch, seq_len, d_model).
    out.hidden_states = FakeTensor(
        np.stack([np.full((n, padded, _HIDDEN), float(i)) for i in range(3)])
    )
    model = MagicMock()
    model.return_value = out

    client = MagicMock()
    client.model = model
    client.tokenizer = tokenizer
    return client


class _FakeKit:
    """esmc_opt stand-in recording calls and printing the kit's lines."""

    def __init__(
        self,
        refuse: str | None = None,
        fallback: list[str] | None = None,
    ) -> None:
        self.refuse = refuse
        self.fallback = fallback or []
        self.calls: list[tuple] = []
        self.mod = ModuleType("esmc_opt")

        class ActivationError(RuntimeError):
            pass

        self.mod.ActivationError = ActivationError  # type: ignore[attr-defined]
        self.mod.enable = self.enable  # type: ignore[attr-defined]
        self.mod.status = self.status  # type: ignore[attr-defined]

    def enable(self, mode: str, **kw: Any) -> dict:
        self.calls.append(("enable", mode, kw))
        if self.refuse:
            print(f"[esmc-opt] NOT ACTIVE: {self.refuse}")
            raise self.mod.ActivationError(self.refuse)  # type: ignore[attr-defined]
        print(f"[esmc-opt] ACTIVE mode={mode} variant={kw.get('variant')}")
        return {"active": True, "mode": mode}

    def status(self) -> dict:
        print("[esmc-opt] APPLIED model#1 levers_applied=pipe,fused")
        return {
            "active": True,
            "mode": "exact",
            "levers_applied": ["pipe", "fused"],
            "levers_fallback": self.fallback,
        }


def _make_esm(client: Any, events: list[str] | None = None) -> dict[str, Any]:
    esmc_mod = ModuleType("esm.models.esmc")
    ESMC = MagicMock()

    def _from_pretrained(name: str, device: Any = None) -> Any:
        if events is not None:
            events.append(f"from_pretrained:{name}:{device}")
        return client

    ESMC.from_pretrained.side_effect = _from_pretrained
    esmc_mod.ESMC = ESMC  # type: ignore[attr-defined]
    return {
        "esm": ModuleType("esm"),
        "esm.models": ModuleType("esm.models"),
        "esm.models.esmc": esmc_mod,
    }


@pytest.fixture(autouse=True)
def _clean_claims():
    _reset_claims()
    _Ctx.entered.clear()
    yield
    _reset_claims()


def _backend(mode: str | None, model_name: str = "Biohub/ESMC-6B", **kw: Any):
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(model_name=model_name, device="cuda", **kw)
    if mode is not None:
        apply_model_opt(backend, ModelOptConfig(mode=mode), "dep")  # type: ignore[arg-type]
    return backend


def _load(backend: Any, client: Any, kit: _FakeKit | None = None, **extra: Any):
    mods: dict[str, Any] = {"torch": _make_torch(), **_make_esm(client, **extra)}
    if kit is not None:
        mods["esmc_opt"] = kit.mod
    with patch.dict(sys.modules, mods):
        backend.load()


def _request(**kw: Any) -> ProteinLanguageRequest:
    return ProteinLanguageRequest(model_name="esmc", sequences=["MKTII", "ACDE"], **kw)


# ---------------------------------------------------------------------------
# Mode support + legacy path
# ---------------------------------------------------------------------------


def test_supported_modes_are_off_and_exact() -> None:
    assert _backend(None).supported_opt_modes() == frozenset({"off", "exact"})


def test_fast_is_rejected_at_apply() -> None:
    from sheaf.backends.esmc import ESMCBackend

    with pytest.raises(ValueError, match="'fast'"):
        apply_model_opt(ESMCBackend(), ModelOptConfig(mode="fast"), "dep")


def test_model_opt_none_uses_transformers_path() -> None:
    backend = _backend(None)
    transformers = ModuleType("transformers")
    transformers.AutoModelForMaskedLM = MagicMock()  # type: ignore[attr-defined]
    transformers.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
    esm_mods = _make_esm(_make_client())
    with patch.dict(sys.modules, {"transformers": transformers, **esm_mods}):
        backend.load()
    esm_mods["esm.models.esmc"].ESMC.from_pretrained.assert_not_called()
    transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once()
    assert backend._client is None


# ---------------------------------------------------------------------------
# mode="off"
# ---------------------------------------------------------------------------


def test_off_loads_through_sdk_without_kit() -> None:
    backend = _backend("off")
    client = _make_client()
    events: list[str] = []
    kit = _FakeKit()
    _load(backend, client, kit, events=events)
    assert events == ["from_pretrained:esmc_6b:device(cuda)"]
    assert kit.calls == []  # off imports / applies nothing of the kit
    assert backend.model_opt_report is None
    client.model.assert_not_called()  # no warm-up under off


def test_sdk_forward_is_the_kits_stock_call() -> None:
    backend = _backend("off")
    client = _make_client()
    _load(backend, client)
    with patch.dict(sys.modules, {"torch": _make_torch()}):
        resp = backend.predict(_request(return_embeddings=True))
    kwargs = client.model.call_args.kwargs
    assert set(kwargs) == {"input_ids", "attention_mask", "output_hidden_states"}
    assert kwargs["output_hidden_states"] is True
    assert (("cuda",), {"dtype": "bf16"}) in _Ctx.entered
    assert resp.seq_lens == [6, 5]
    assert len(resp.logits[0]) == 6 and len(resp.logits[1]) == 5
    # Embeddings = the last entry of the stacked hidden_states tensor.
    assert resp.embeddings[0][0] == [2.0] * _HIDDEN
    assert resp.hidden_dim == _HIDDEN


def test_sdk_path_hidden_states_iterate_stacked_tensor() -> None:
    backend = _backend("off")
    _load(backend, _make_client())
    with patch.dict(sys.modules, {"torch": _make_torch()}):
        resp = backend.predict(_request(output_hidden_states=True))
    assert len(resp.hidden_states) == 3
    assert resp.hidden_states[1][0][0] == [1.0] * _HIDDEN


# ---------------------------------------------------------------------------
# mode="exact"
# ---------------------------------------------------------------------------


def test_exact_enables_kit_before_client_and_warms_up(caplog) -> None:
    caplog.set_level(logging.INFO, logger="sheaf.model_opt")
    backend = _backend("exact", model_name="biohub/ESMC-600M")
    client = _make_client()
    kit = _FakeKit()
    events: list[str] = []
    real_enable = kit.enable

    def _enable(mode: str, **kw: Any) -> dict:
        events.append("enable")
        return real_enable(mode, **kw)

    kit.mod.enable = _enable  # type: ignore[attr-defined]
    _load(backend, client, kit, events=events)

    assert events == ["enable", "from_pretrained:esmc_600m:device(cuda)"]
    assert kit.calls == [
        ("enable", "exact", {"variant": "600m", "regime": "batched", "strict": True})
    ]
    client.model.assert_called_once()  # the load-time warm-up forward
    assert backend.model_opt_report["levers_applied"] == ["pipe", "fused"]
    assert backend.model_opt_lines == [
        "[esmc-opt] ACTIVE mode=exact variant=600m",
        "[esmc-opt] APPLIED model#1 levers_applied=pipe,fused",
    ]
    logged = [r.getMessage() for r in caplog.records if r.name == "sheaf.model_opt"]
    assert "[esmc-opt] ACTIVE mode=exact variant=600m" in logged


def test_exact_refusal_at_enable_raises_at_load() -> None:
    backend = _backend("exact")
    client = _make_client()
    kit = _FakeKit(refuse="no visible GPU")
    with pytest.raises(ModelOptNotActiveError, match="NOT ACTIVE: no visible GPU"):
        _load(backend, client, kit)
    assert "[esmc-opt] NOT ACTIVE: no visible GPU" in backend.model_opt_lines
    client.model.assert_not_called()


def test_exact_partial_activation_at_from_pretrained_raises() -> None:
    backend = _backend("exact")
    kit = _FakeKit()
    esm_mods = _make_esm(_make_client())
    esm_mods["esm.models.esmc"].ESMC.from_pretrained.side_effect = (
        kit.mod.ActivationError("partial activation — fused: nvcc missing")  # type: ignore[attr-defined]
    )
    with patch.dict(
        sys.modules, {"torch": _make_torch(), "esmc_opt": kit.mod, **esm_mods}
    ):
        with pytest.raises(ModelOptNotActiveError, match="partial activation"):
            backend.load()


def test_exact_non_kit_error_at_from_pretrained_propagates() -> None:
    backend = _backend("exact")
    kit = _FakeKit()
    esm_mods = _make_esm(_make_client())
    esm_mods["esm.models.esmc"].ESMC.from_pretrained.side_effect = OSError("404")
    with patch.dict(
        sys.modules, {"torch": _make_torch(), "esmc_opt": kit.mod, **esm_mods}
    ):
        with pytest.raises(OSError, match="404"):
            backend.load()


def test_exact_levers_fallback_after_warmup_raises() -> None:
    backend = _backend("exact")
    kit = _FakeKit(fallback=["fused"])
    with pytest.raises(ModelOptNotActiveError, match="levers_fallback=\\['fused'\\]"):
        _load(backend, _make_client(), kit)


def test_exact_without_kit_installed_raises() -> None:
    backend = _backend("exact")
    real_import = __import__

    def _no_kit(name: str, *a: Any, **kw: Any) -> Any:
        if name == "esmc_opt":
            raise ImportError("No module named 'esmc_opt'")
        return real_import(name, *a, **kw)

    with patch("builtins.__import__", side_effect=_no_kit):
        with pytest.raises(ModelOptNotActiveError, match="not installed"):
            _load(backend, _make_client())


# ---------------------------------------------------------------------------
# Validation on the model_opt path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "sdk"),
    [
        ("Biohub/ESMC-6B", "esmc_6b"),
        ("biohub/ESMC-300M", "esmc_300m"),
        ("esmc_600m", "esmc_600m"),
    ],
)
def test_model_name_maps_to_sdk_name(name: str, sdk: str) -> None:
    backend = _backend("off", model_name=name)
    events: list[str] = []
    _load(backend, _make_client(), events=events)
    assert events == [f"from_pretrained:{sdk}:device(cuda)"]


def test_unknown_model_name_rejected() -> None:
    backend = _backend("off", model_name="someone/ESMC-fork")
    with pytest.raises(ValueError, match="ESM C kit's checkpoints"):
        _load(backend, _make_client())


def test_device_map_rejected() -> None:
    backend = _backend("off", device_map="auto")
    with pytest.raises(ValueError, match="device_map"):
        _load(backend, _make_client())


def test_levers_off_rejected() -> None:
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend()
    apply_model_opt(backend, ModelOptConfig(mode="exact", levers_off=["fused"]), "dep")
    with pytest.raises(ValueError, match="levers_off must be empty"):
        _load(backend, _make_client(), _FakeKit())


def test_second_mode_in_same_process_refused() -> None:
    first = _backend("exact")
    _load(first, _make_client(), _FakeKit())
    second = _backend("off")
    with pytest.raises(ModelOptNotActiveError, match="already claimed"):
        _load(second, _make_client())
