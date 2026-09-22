"""Tests for sheaf.model_opt — spec plumbing, isolation guards, kit line capture.

No GPU, torch or kit required: every kit interaction is faked.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest
from pydantic import ValidationError

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends.base import ModelBackend
from sheaf.batch.spec import BatchSpec, JsonlSink, JsonlSource
from sheaf.model_opt import (
    ModelOptConfig,
    ModelOptNotActiveError,
    _reset_claims,
    apply_model_opt,
    capture_kit_lines,
    claim_process,
    configure_jit_env,
)
from sheaf.spec import ModelSpec


class _PlainBackend(ModelBackend):
    """A backend that supports no kit (the default)."""

    def load(self) -> None:
        pass

    def predict(self, request: BaseRequest) -> BaseResponse:
        raise NotImplementedError

    @property
    def model_type(self) -> str:
        return ModelType.PROTEIN_LANGUAGE


class _KitBackend(_PlainBackend):
    def supported_opt_modes(self) -> frozenset[str]:
        return frozenset({"off", "exact"})


@pytest.fixture(autouse=True)
def _clean_claims():
    _reset_claims()
    yield
    _reset_claims()


# ---------------------------------------------------------------------------
# ModelOptConfig / ModelSpec
# ---------------------------------------------------------------------------


def test_model_spec_default_is_none() -> None:
    spec = ModelSpec(name="m", model_type=ModelType.PROTEIN_LANGUAGE, backend="esmc")
    assert spec.model_opt is None


def test_model_opt_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        ModelOptConfig(mode="turbo")  # type: ignore[arg-type]


def test_model_opt_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ModelOptConfig(mode="exact", nope=1)  # type: ignore[call-arg]


def test_model_spec_round_trips_model_opt() -> None:
    spec = ModelSpec(
        name="m",
        model_type=ModelType.PROTEIN_LANGUAGE,
        backend="esmc",
        model_opt=ModelOptConfig(mode="exact", jit_root="/jit"),
    )
    again = ModelSpec.model_validate(spec.model_dump())
    assert again.model_opt == spec.model_opt


# ---------------------------------------------------------------------------
# apply_model_opt
# ---------------------------------------------------------------------------


def test_apply_none_is_noop() -> None:
    backend = _PlainBackend()
    apply_model_opt(backend, None, "m")
    assert not hasattr(backend, "_model_opt")


def test_apply_rejects_backend_without_kit() -> None:
    with pytest.raises(ValueError, match="supports no inference optimization kit"):
        apply_model_opt(_PlainBackend(), ModelOptConfig(mode="exact"), "m")


def test_apply_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="'fast'"):
        apply_model_opt(_KitBackend(), ModelOptConfig(mode="fast"), "m")


def test_apply_hands_config_to_backend() -> None:
    backend = _KitBackend()
    cfg = ModelOptConfig(mode="exact")
    apply_model_opt(backend, cfg, "m")
    assert backend._model_opt is cfg


# ---------------------------------------------------------------------------
# claim_process
# ---------------------------------------------------------------------------


def test_claim_same_mode_twice_is_allowed() -> None:
    claim_process("esmc", "exact", "a")
    claim_process("esmc", "exact", "b")


def test_claim_different_mode_raises() -> None:
    claim_process("esmc", "exact", "a")
    with pytest.raises(ModelOptNotActiveError, match="already claimed by 'a'"):
        claim_process("esmc", "off", "b")


def test_claims_are_per_kit() -> None:
    claim_process("esmc", "exact", "a")
    claim_process("esmfold2", "fast", "b")


# ---------------------------------------------------------------------------
# Isolation: ModalServer + BatchSpec
# ---------------------------------------------------------------------------


def _spec(name: str, model_opt: ModelOptConfig | None = None) -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.PROTEIN_LANGUAGE,
        backend="esmc",
        model_opt=model_opt,
    )


def test_modal_rejects_model_opt_spec_sharing_container() -> None:
    from sheaf.modal_server import _check_model_opt_isolation

    with pytest.raises(ValueError, match="own ModalServer"):
        _check_model_opt_isolation(
            [_spec("a", ModelOptConfig(mode="exact")), _spec("b")]
        )


def test_modal_allows_single_model_opt_spec() -> None:
    from sheaf.modal_server import _check_model_opt_isolation

    _check_model_opt_isolation([_spec("a", ModelOptConfig(mode="exact"))])
    _check_model_opt_isolation([_spec("a"), _spec("b")])


def _batch_spec(**kw) -> BatchSpec:
    return BatchSpec(
        name="b",
        model_type=ModelType.PROTEIN_LANGUAGE,
        backend="esmc",
        source=JsonlSource(path="in.jsonl"),
        sink=JsonlSink(path="out.jsonl"),
        **kw,
    )


def test_batch_spec_model_opt_requires_actors() -> None:
    with pytest.raises(ValidationError, match="compute='actors'"):
        _batch_spec(model_opt=ModelOptConfig(mode="exact"))


def test_batch_spec_model_opt_with_actors_ok() -> None:
    spec = _batch_spec(
        model_opt=ModelOptConfig(mode="exact"), compute="actors", num_actors=1
    )
    assert spec.model_opt is not None


# ---------------------------------------------------------------------------
# configure_jit_env
# ---------------------------------------------------------------------------


_JIT_VARS = (
    "MODEL_OPT_JIT_ROOT",
    "MODEL_OPT_STACK_KEY",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
)


def test_jit_env_noop_without_root(monkeypatch) -> None:
    for v in _JIT_VARS:
        monkeypatch.delenv(v, raising=False)
    assert configure_jit_env(ModelOptConfig(mode="exact"), "k") == {}
    assert "TRITON_CACHE_DIR" not in os.environ


def test_jit_env_sets_keyed_private_dirs(monkeypatch, tmp_path) -> None:
    for v in _JIT_VARS:
        monkeypatch.delenv(v, raising=False)
    root = tmp_path / "jit"
    cfg = ModelOptConfig(mode="exact", jit_root=str(root))
    set_now = configure_jit_env(cfg, "torch2.11.0-cu130-sm90")
    triton = root / "torch2.11.0-cu130-sm90" / "triton"
    assert set_now["TRITON_CACHE_DIR"] == str(triton)
    assert os.environ["MODEL_OPT_STACK_KEY"] == "torch2.11.0-cu130-sm90"
    assert triton.is_dir()
    assert (triton.stat().st_mode & 0o077) == 0


def test_jit_env_preset_vars_win(monkeypatch, tmp_path) -> None:
    for v in _JIT_VARS:
        monkeypatch.delenv(v, raising=False)
    preset = tmp_path / "mine"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(preset))
    cfg = ModelOptConfig(mode="exact", jit_root=str(tmp_path / "jit"))
    set_now = configure_jit_env(cfg, "k")
    assert "TRITON_CACHE_DIR" not in set_now
    assert os.environ["TRITON_CACHE_DIR"] == str(preset)


# ---------------------------------------------------------------------------
# capture_kit_lines
# ---------------------------------------------------------------------------


def test_capture_relogs_only_kit_lines(caplog, capsys) -> None:
    caplog.set_level(logging.INFO, logger="sheaf.model_opt")
    with capture_kit_lines("esmc-opt", "dep") as lines:
        print("[esmc-opt] ACTIVE mode=exact variant=6b")
        print("unrelated chatter")
        sys.stderr.write("[esmc-opt] APPLIED model#1 levers_applied=pipe,fused\n")
    assert lines == [
        "[esmc-opt] ACTIVE mode=exact variant=6b",
        "[esmc-opt] APPLIED model#1 levers_applied=pipe,fused",
    ]
    logged = [r for r in caplog.records if r.name == "sheaf.model_opt"]
    assert [r.getMessage() for r in logged] == lines
    assert all(r.kit == "esmc-opt" and r.deployment == "dep" for r in logged)
    # Output still reaches the real streams.
    out = capsys.readouterr()
    assert "unrelated chatter" in out.out
    assert "APPLIED" in out.err


def test_capture_collects_lines_even_when_block_raises() -> None:
    sink: list[str] = []
    with pytest.raises(RuntimeError):
        with capture_kit_lines("esmc-opt", "dep", sink):
            print("[esmc-opt] NOT ACTIVE: no visible GPU")
            raise RuntimeError("boom")
    assert sink == ["[esmc-opt] NOT ACTIVE: no visible GPU"]
