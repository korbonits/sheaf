"""Tests for MolFormerBackend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes model_name and trust_remote_code to both from_pretrained calls
  - load() stores tokenizer for test injectability
  - load() moves model to specified device
  - predict() rejects non-SmallMoleculeRequest inputs
  - predict() returns SmallMoleculeResponse with correct structure
  - predict() dim matches hidden_size of fake model output
  - predict() mean pooling uses attention mask to exclude padding
  - predict() cls pooling takes the first token
  - predict() normalize=True produces unit-norm embeddings
  - predict() normalize=False returns raw embeddings
  - predict() tokenizer called with full smiles list and padding=True
  - predict() one embedding per molecule
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import builtins
import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.small_molecule import SmallMoleculeRequest, SmallMoleculeResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 768  # MoLFormer-XL
_N_MOL = 2  # molecules per batch
_SEQ_LEN = 12  # token length (including CLS and padding)

# ---------------------------------------------------------------------------
# FakeTensor — numpy-backed; supports all ops used in MolFormerBackend._run()
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: list | np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(self._data[key])  # type: ignore[index]

    def unsqueeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self._data, axis=dim))

    def float(self) -> FakeTensor:
        return self

    def __mul__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self._data * other._data)

    def __rmul__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(other._data * self._data)

    def sum(self, dim: int) -> FakeTensor:
        return FakeTensor(self._data.sum(axis=dim))

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self._data / other._data)

    def mean(self, dim: int) -> FakeTensor:
        return FakeTensor(self._data.mean(axis=dim))

    def norm(self, dim: int = -1, keepdim: bool = False) -> FakeTensor:
        return FakeTensor(np.linalg.norm(self._data, axis=dim, keepdims=keepdim))

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list:
        return self._data.tolist()

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Fake tokenizer output — dict-like with .items() and key access
# ---------------------------------------------------------------------------


class _FakeTokenizerOutput:
    """Wraps two FakeTensors in a dict-like object with .items()."""

    def __init__(self, n: int = _N_MOL, seq_len: int = _SEQ_LEN) -> None:
        # All positions real (no padding) for simple test arithmetic
        self._d = {
            "input_ids": FakeTensor(np.ones((n, seq_len), dtype=np.float32)),
            "attention_mask": FakeTensor(np.ones((n, seq_len), dtype=np.float32)),
        }

    def items(self):  # type: ignore[no-untyped-def]
        return self._d.items()


# ---------------------------------------------------------------------------
# Fake torch module
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _make_torch_mod() -> ModuleType:
    mod = ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]
    return mod


_torch_mod = _make_torch_mod()

# ---------------------------------------------------------------------------
# Fake transformers module factory
# ---------------------------------------------------------------------------


def _make_model_output(
    n: int = _N_MOL, seq_len: int = _SEQ_LEN, hidden_size: int = _HIDDEN_DIM
) -> MagicMock:
    hidden = FakeTensor(np.ones((n, seq_len, hidden_size), dtype=np.float32))
    out = MagicMock()
    out.last_hidden_state = hidden
    return out


def _make_transformers_mod(
    n: int = _N_MOL,
    seq_len: int = _SEQ_LEN,
    hidden_size: int = _HIDDEN_DIM,
) -> ModuleType:
    model_output = _make_model_output(n=n, seq_len=seq_len, hidden_size=hidden_size)
    model = MagicMock()
    model.return_value = model_output
    model.to.return_value = model
    model.eval.return_value = None

    tokenizer = MagicMock()
    tokenizer.return_value = _FakeTokenizerOutput(n=n, seq_len=seq_len)

    mod = ModuleType("transformers")
    mod.AutoModel = MagicMock()  # type: ignore[attr-defined]
    mod.AutoModel.from_pretrained.return_value = model  # type: ignore[attr-defined]
    mod.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
    mod.AutoTokenizer.from_pretrained.return_value = tokenizer  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.molformer import MolFormerBackend

    backend = MolFormerBackend(model_name="ibm/MoLFormer-XL-both-10pct", device="cpu")
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    model = mod.AutoModel.from_pretrained.return_value
    tokenizer = mod.AutoTokenizer.from_pretrained.return_value
    backend._model = model
    backend._tokenizer = tokenizer
    return model, tokenizer


# ---------------------------------------------------------------------------
# Request factory
# ---------------------------------------------------------------------------

_ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def _make_request(
    smiles: list[str] | None = None,
    pooling: str = "mean",
    normalize: bool = False,
) -> SmallMoleculeRequest:
    if smiles is None:
        smiles = [_ASPIRIN, _CAFFEINE]
    return SmallMoleculeRequest(
        model_name="molformer",
        smiles=smiles,
        pooling=pooling,  # type: ignore[arg-type]
        normalize=normalize,
    )


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    from sheaf.backends.molformer import MolFormerBackend

    backend = MolFormerBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[small-molecule\\]"),
    ):
        backend.load()


def test_load_passes_model_name_and_trust_remote_code(
    mock_transformers: ModuleType,
) -> None:
    from sheaf.backends.molformer import MolFormerBackend

    model_name = "ibm/MoLFormer-XL-both-100pct"
    backend = MolFormerBackend(model_name=model_name)
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        model_name, trust_remote_code=True
    )
    mock_transformers.AutoModel.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        model_name, trust_remote_code=True
    )


def test_load_stores_tokenizer(mock_transformers: ModuleType) -> None:
    from sheaf.backends.molformer import MolFormerBackend

    backend = MolFormerBackend()
    assert backend._tokenizer is None
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()
    assert backend._tokenizer is not None


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.molformer import MolFormerBackend

    backend = MolFormerBackend(device="cuda")
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    mock_transformers.AutoModel.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.genomic import GenomicRequest

    req = GenomicRequest(model_name="x", sequences=["ATCG"])
    with pytest.raises(TypeError, match="SmallMoleculeRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_small_molecule_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert isinstance(resp, SmallMoleculeResponse)
    assert resp.dim == _HIDDEN_DIM
    assert len(resp.embeddings) == _N_MOL


def test_predict_dim_matches_hidden_size(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    hidden_size = 256
    mock = _make_transformers_mod(hidden_size=hidden_size)
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert resp.dim == hidden_size
    assert all(len(e) == hidden_size for e in resp.embeddings)


# ---------------------------------------------------------------------------
# predict() — pooling
# ---------------------------------------------------------------------------


def test_predict_mean_pooling_uses_attention_mask(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Mean pooling gives (1 * real_tokens + 0 * padding) / real_token_count.

    With all-ones hidden states and all-ones mask, the mean is 1.0.
    """
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(pooling="mean"))

    # All-ones hidden + all-ones mask → mean = 1.0
    assert resp.embeddings[0][0] == pytest.approx(1.0)


def test_predict_cls_pooling_takes_first_token(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """CLS pooling returns position-0 hidden state; other positions differ."""
    n, seq_len, hidden_size = 1, 6, 4
    hidden_data = np.full((n, seq_len, hidden_size), 99.0, dtype=np.float32)
    hidden_data[0, 0, :] = 7.0  # CLS — selected
    model_output = MagicMock()
    model_output.last_hidden_state = FakeTensor(hidden_data)

    mock = _make_transformers_mod(n=n, seq_len=seq_len, hidden_size=hidden_size)
    model, tokenizer = _wire(loaded_backend, mock)
    model.return_value = model_output
    tokenizer.return_value = _FakeTokenizerOutput(n=n, seq_len=seq_len)

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(smiles=[_ASPIRIN], pooling="cls"))

    assert all(v == pytest.approx(7.0) for v in resp.embeddings[0])


# ---------------------------------------------------------------------------
# predict() — normalization
# ---------------------------------------------------------------------------


def test_predict_normalize_true_unit_norm(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(normalize=True))

    for emb in resp.embeddings:
        norm = math.sqrt(sum(x**2 for x in emb))
        assert abs(norm - 1.0) < 1e-5


def test_predict_normalize_false_raw(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """normalize=False keeps raw embedding magnitude."""
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(normalize=False))

    # All-ones embedding → L2 norm = sqrt(768) ≈ 27.7, not 1.0
    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 1.0) > 1.0


# ---------------------------------------------------------------------------
# predict() — tokenizer call and batch shape
# ---------------------------------------------------------------------------


def test_predict_tokenizer_called_with_smiles_and_padding(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    """Tokenizer receives the full SMILES list with padding=True."""
    mock = _make_transformers_mod()
    model, tokenizer = _wire(loaded_backend, mock)
    smiles = [_ASPIRIN, _CAFFEINE]

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(_make_request(smiles=smiles))

    tokenizer.assert_called_once_with(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


def test_predict_one_embedding_per_molecule(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """N molecules → N embeddings, each of length dim."""
    n = 4
    mock = _make_transformers_mod(n=n)
    _wire(loaded_backend, mock)
    smiles = [_ASPIRIN, _CAFFEINE, "C", "CC"]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(smiles=smiles))

    assert len(resp.embeddings) == n
    assert all(len(e) == _HIDDEN_DIM for e in resp.embeddings)
    assert resp.dim == _HIDDEN_DIM


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_independently(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod(n=1)
    model, tokenizer = _wire(loaded_backend, mock)
    tokenizer.return_value = _FakeTokenizerOutput(n=1)

    reqs = [
        _make_request(smiles=[_ASPIRIN]),
        _make_request(smiles=[_CAFFEINE]),
    ]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, SmallMoleculeResponse) for r in responses)
    assert model.call_count == 2
