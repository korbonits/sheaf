"""Tests for ESM3Backend — fully mocked, no esm or torch required.

Covers:
  - load() raises ImportError when esm is absent
  - load() passes correct model_name and moves model to device
  - predict() rejects non-MolecularRequest inputs
  - predict() mean pooling — mean of residue tokens (positions 1:-1)
  - predict() cls pooling — BOS token at position 0
  - predict() normalize=True — unit-norm output
  - predict() normalize=False — raw embeddings returned
  - predict() multiple sequences in one request — independent forward passes
  - predict() returns MolecularResponse with correct shape
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.molecular import MolecularRequest, MolecularResponse

# ---------------------------------------------------------------------------
# FakeTensor — numpy-backed, supports ESM-3 hidden-state ops
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.array(data, dtype=np.float32)

    def norm(self, dim: int = -1, keepdim: bool = False) -> FakeTensor:
        return FakeTensor(np.linalg.norm(self._data, axis=dim, keepdims=keepdim))

    def mean(self, dim: int = 0) -> FakeTensor:
        return FakeTensor(self._data.mean(axis=dim))

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self._data / other._data)

    def __getitem__(self, idx: object) -> FakeTensor:
        return FakeTensor(self._data[idx])  # type: ignore[index]

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def tolist(self) -> list:  # type: ignore[override]
        return self._data.tolist()

    @property
    def shape(self) -> tuple:
        return self._data.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fake torch module
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_) -> None:  # type: ignore[no-untyped-def]
        pass


def _make_torch_mod() -> ModuleType:
    mod = ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]
    return mod


_torch_mod = _make_torch_mod()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 1536  # esm3-sm-open-v1
_SEQ_LEN = 7  # 5 residues + BOS + EOS


# ---------------------------------------------------------------------------
# Fake esm module factory
# ---------------------------------------------------------------------------


def _make_esm_mod(
    hidden_dim: int = _HIDDEN_DIM,
    seq_len: int = _SEQ_LEN,
) -> tuple[ModuleType, MagicMock, MagicMock]:
    """Return (esm_pkg, model_mock, ESMProtein_cls_mock)."""
    hidden = FakeTensor(np.random.randn(1, seq_len, hidden_dim).astype(np.float32))

    output = MagicMock()
    output.embeddings = hidden

    # protein_tensor.sequence is a MagicMock; .unsqueeze(0) returns a MagicMock
    protein_tensor = MagicMock()

    model = MagicMock()
    model.to.return_value = model
    model.encode.return_value = protein_tensor
    model.forward.return_value = output

    esm3_cls = MagicMock()
    esm3_cls.from_pretrained.return_value = model

    protein_cls = MagicMock()

    # Build the fake module tree
    esm_mod = ModuleType("esm")
    models_mod = ModuleType("esm.models")
    esm3_mod = ModuleType("esm.models.esm3")
    sdk_mod = ModuleType("esm.sdk")
    api_mod = ModuleType("esm.sdk.api")

    esm3_mod.ESM3 = esm3_cls  # type: ignore[attr-defined]
    api_mod.ESMProtein = protein_cls  # type: ignore[attr-defined]

    esm_mod.models = models_mod  # type: ignore[attr-defined]
    esm_mod.sdk = sdk_mod  # type: ignore[attr-defined]
    models_mod.esm3 = esm3_mod  # type: ignore[attr-defined]
    sdk_mod.api = api_mod  # type: ignore[attr-defined]

    return esm_mod, model, protein_cls


def _esm_sys_modules(esm_mod: ModuleType) -> dict[str, ModuleType]:
    return {
        "esm": esm_mod,
        "esm.models": esm_mod.models,  # type: ignore[attr-defined]
        "esm.models.esm3": esm_mod.models.esm3,  # type: ignore[attr-defined]
        "esm.sdk": esm_mod.sdk,  # type: ignore[attr-defined]
        "esm.sdk.api": esm_mod.sdk.api,  # type: ignore[attr-defined]
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def esm_mod_and_model():  # type: ignore[no-untyped-def]
    return _make_esm_mod()


@pytest.fixture
def loaded_backend(esm_mod_and_model):  # type: ignore[no-untyped-def]
    from sheaf.backends.esm3 import ESM3Backend

    esm_mod, model, protein_cls = esm_mod_and_model
    backend = ESM3Backend(model_name="esm3-sm-open-v1", device="cpu")
    with patch.dict(sys.modules, {**_esm_sys_modules(esm_mod), "torch": _torch_mod}):
        backend.load()
    # Wire internals directly for test assertions
    backend._model = model
    backend._ESMProtein = protein_cls
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_esm() -> None:
    import builtins

    from sheaf.backends.esm3 import ESM3Backend

    backend = ESM3Backend()
    mods_without = {k: v for k, v in sys.modules.items() if not k.startswith("esm")}

    _real_import = builtins.__import__

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "esm" or name.startswith("esm."):
            raise ModuleNotFoundError("No module named 'esm'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[molecular\\]"),
    ):
        backend.load()


def test_load_passes_model_name_and_device() -> None:
    from sheaf.backends.esm3 import ESM3Backend

    backend = ESM3Backend(model_name="esm3-sm-open-v1", device="cuda")
    esm_mod, model, _ = _make_esm_mod()
    with patch.dict(sys.modules, {**_esm_sys_modules(esm_mod), "torch": _torch_mod}):
        backend.load()

    esm_mod.models.esm3.ESM3.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "esm3-sm-open-v1"
    )
    model.to.assert_called_once_with("cuda")


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_request_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="MolecularRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — pooling
# ---------------------------------------------------------------------------


def test_predict_mean_pooling(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    """Mean pooling averages positions 1:-1, ignoring BOS (0) and EOS (-1)."""
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls
    loaded_backend._pooling = "mean"

    # BOS = 99, residues = 2.0 (constant), EOS = 99 → mean of residues = 2.0
    hidden_data = np.full((1, _SEQ_LEN, _HIDDEN_DIM), 2.0, dtype=np.float32)
    hidden_data[0, 0, :] = 99.0  # BOS — excluded
    hidden_data[0, -1, :] = 99.0  # EOS — excluded
    output = MagicMock()
    output.embeddings = FakeTensor(hidden_data)
    model.forward.return_value = output

    req = MolecularRequest(
        model_name="esm3", sequences=["MKTII"], pooling="mean", normalize=False
    )
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.embeddings[0][0] == pytest.approx(2.0)


def test_predict_cls_pooling(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    """CLS pooling uses BOS token at position 0."""
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls
    loaded_backend._pooling = "cls"

    known = np.zeros((_HIDDEN_DIM,), dtype=np.float32)
    known[0] = 5.0
    hidden_data = np.zeros((1, _SEQ_LEN, _HIDDEN_DIM), dtype=np.float32)
    hidden_data[0, 0, :] = known
    hidden_data[0, 1:, :] = 99.0  # residues + EOS — ignored
    output = MagicMock()
    output.embeddings = FakeTensor(hidden_data)
    model.forward.return_value = output

    req = MolecularRequest(
        model_name="esm3", sequences=["MKTII"], pooling="cls", normalize=False
    )
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.embeddings[0][0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# predict() — normalization
# ---------------------------------------------------------------------------


def test_predict_normalize_true(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls

    req = MolecularRequest(model_name="esm3", sequences=["MKTII"], normalize=True)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 1.0) < 1e-5


def test_predict_normalize_false(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls
    loaded_backend._pooling = "cls"

    # CLS token: [3, 4, 0, ...] → norm = 5
    hidden_data = np.zeros((1, _SEQ_LEN, _HIDDEN_DIM), dtype=np.float32)
    hidden_data[0, 0, 0] = 3.0
    hidden_data[0, 0, 1] = 4.0
    output = MagicMock()
    output.embeddings = FakeTensor(hidden_data)
    model.forward.return_value = output

    req = MolecularRequest(
        model_name="esm3", sequences=["MKTII"], pooling="cls", normalize=False
    )
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# predict() — response shape
# ---------------------------------------------------------------------------


def test_predict_returns_molecular_response(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls

    req = MolecularRequest(model_name="esm3", sequences=["MKTII"])
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, MolecularResponse)
    assert resp.dim == _HIDDEN_DIM
    assert len(resp.embeddings) == 1
    assert len(resp.embeddings[0]) == _HIDDEN_DIM


def test_predict_multiple_sequences(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    """Multiple sequences in one request → one forward pass per sequence."""
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls

    seqs = ["MKTII", "ACDEF", "LMNPQ"]
    req = MolecularRequest(model_name="esm3", sequences=seqs)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert len(resp.embeddings) == 3
    assert resp.dim == _HIDDEN_DIM
    assert model.encode.call_count == 3
    assert model.forward.call_count == 3


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend, esm_mod_and_model) -> None:  # type: ignore[no-untyped-def]
    _, model, protein_cls = esm_mod_and_model
    loaded_backend._model = model
    loaded_backend._ESMProtein = protein_cls

    reqs = [
        MolecularRequest(model_name="esm3", sequences=["MKTII"]),
        MolecularRequest(model_name="esm3", sequences=["ACDEF"]),
    ]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, MolecularResponse) for r in responses)
    assert model.encode.call_count == 2
