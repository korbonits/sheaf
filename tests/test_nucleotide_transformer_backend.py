"""Tests for NucleotideTransformerBackend — fully mocked, no transformers required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes model_name to AutoTokenizer and AutoModel
  - load() moves model to specified device
  - load() stores tokenizer for test injectability
  - predict() rejects non-GenomicRequest inputs
  - predict() returns GenomicResponse with correct structure
  - predict() dim matches hidden_size of fake model output
  - predict() mean pooling excludes CLS and EOS special tokens
  - predict() cls pooling takes the first token
  - predict() normalize=True produces unit-norm embeddings
  - predict() normalize=False returns raw embeddings
  - predict() calls tokenizer once per sequence
  - predict() multiple sequences returns one embedding per sequence
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import builtins
import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from sheaf.api.genomic import GenomicRequest, GenomicResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 512  # NT-v2-100M hidden size
_SEQ_LEN = 10  # token length including CLS and EOS

# ---------------------------------------------------------------------------
# FakeTensor — numpy-backed; supports slicing, mean, norm, truediv, to()
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: list | np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(self._data[key])  # type: ignore[index]

    def mean(self, dim: int) -> FakeTensor:
        return FakeTensor(self._data.mean(axis=dim))

    def norm(self, dim: int = -1, keepdim: bool = False) -> FakeTensor:
        return FakeTensor(np.linalg.norm(self._data, axis=dim, keepdims=keepdim))

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self._data / other._data)

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def tolist(self) -> list:
        return self._data.tolist()

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Fake tokenizer output — dict-like with .items()
# ---------------------------------------------------------------------------


class _FakeTokenizerOutput:
    def __init__(self) -> None:
        self._d = {
            "input_ids": FakeTensor(np.ones((1, _SEQ_LEN), dtype=np.float32)),
            "attention_mask": FakeTensor(np.ones((1, _SEQ_LEN), dtype=np.float32)),
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
    n_tokens: int = _SEQ_LEN, hidden_size: int = _HIDDEN_DIM
) -> MagicMock:
    hidden = FakeTensor(np.ones((1, n_tokens, hidden_size), dtype=np.float32))
    output = MagicMock()
    output.last_hidden_state = hidden
    return output


def _make_transformers_mod(
    hidden_size: int = _HIDDEN_DIM,
    n_tokens: int = _SEQ_LEN,
) -> ModuleType:
    model_output = _make_model_output(n_tokens=n_tokens, hidden_size=hidden_size)
    model = MagicMock()
    model.return_value = model_output
    model.to.return_value = model
    model.eval.return_value = None

    tokenizer = MagicMock()
    tokenizer.return_value = _FakeTokenizerOutput()

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
    from sheaf.backends.nucleotide_transformer import NucleotideTransformerBackend

    backend = NucleotideTransformerBackend(
        model_name="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        device="cpu",
    )
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


def _make_request(
    sequences: list[str] | None = None,
    pooling: str = "mean",
    normalize: bool = False,
) -> GenomicRequest:
    if sequences is None:
        sequences = ["ATCGATCGATCG"]
    return GenomicRequest(
        model_name="nt",
        sequences=sequences,
        pooling=pooling,  # type: ignore[arg-type]
        normalize=normalize,
    )


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    from sheaf.backends.nucleotide_transformer import NucleotideTransformerBackend

    backend = NucleotideTransformerBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[genomics\\]"),
    ):
        backend.load()


def test_load_passes_model_name_to_tokenizer_and_model(
    mock_transformers: ModuleType,
) -> None:
    from sheaf.backends.nucleotide_transformer import NucleotideTransformerBackend

    model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    backend = NucleotideTransformerBackend(model_name=model_name)
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        model_name
    )
    mock_transformers.AutoModel.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        model_name
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.nucleotide_transformer import NucleotideTransformerBackend

    backend = NucleotideTransformerBackend(device="cuda")
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    mock_transformers.AutoModel.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


def test_load_stores_tokenizer(mock_transformers: ModuleType) -> None:
    from sheaf.backends.nucleotide_transformer import NucleotideTransformerBackend

    backend = NucleotideTransformerBackend()
    assert backend._tokenizer is None
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    assert backend._tokenizer is not None


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.molecular import MolecularRequest

    req = MolecularRequest(model_name="x", sequences=["MKTII"])
    with pytest.raises(TypeError, match="GenomicRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_genomic_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert isinstance(resp, GenomicResponse)
    assert resp.dim == _HIDDEN_DIM
    assert len(resp.embeddings) == 1
    assert len(resp.embeddings[0]) == _HIDDEN_DIM


def test_predict_dim_matches_hidden_size(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    hidden_size = 256
    mock = _make_transformers_mod(hidden_size=hidden_size)
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert resp.dim == hidden_size
    assert len(resp.embeddings[0]) == hidden_size


# ---------------------------------------------------------------------------
# predict() — pooling
# ---------------------------------------------------------------------------


def test_predict_mean_pooling_excludes_special_tokens(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    """Mean pooling excludes CLS (pos 0) and EOS (pos -1); rest are 2.0."""
    n_tokens, hidden_size = 8, 4
    hidden_data = np.full((1, n_tokens, hidden_size), 2.0, dtype=np.float32)
    hidden_data[0, 0, :] = 99.0  # CLS — excluded
    hidden_data[0, -1, :] = 99.0  # EOS — excluded
    model_output = MagicMock()
    model_output.last_hidden_state = FakeTensor(hidden_data)

    mock = _make_transformers_mod(n_tokens=n_tokens, hidden_size=hidden_size)
    model, tokenizer = _wire(loaded_backend, mock)
    model.return_value = model_output

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(pooling="mean"))

    assert resp.embeddings[0][0] == pytest.approx(2.0)


def test_predict_cls_pooling_takes_first_token(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    """CLS pooling returns position-0 token; rest are 0."""
    n_tokens, hidden_size = 6, 4
    hidden_data = np.zeros((1, n_tokens, hidden_size), dtype=np.float32)
    hidden_data[0, 0, :] = 3.0  # CLS token — selected
    hidden_data[0, 1:, :] = 99.0  # all others — ignored
    model_output = MagicMock()
    model_output.last_hidden_state = FakeTensor(hidden_data)

    mock = _make_transformers_mod(n_tokens=n_tokens, hidden_size=hidden_size)
    model, tokenizer = _wire(loaded_backend, mock)
    model.return_value = model_output

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(pooling="cls"))

    assert resp.embeddings[0][0] == pytest.approx(3.0)
    assert all(v == pytest.approx(3.0) for v in resp.embeddings[0])


# ---------------------------------------------------------------------------
# predict() — normalization
# ---------------------------------------------------------------------------


def test_predict_normalize_true_unit_norm(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(normalize=True))

    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 1.0) < 1e-5


def test_predict_normalize_false_raw(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """normalize=False returns embeddings with non-unit norm."""
    n_tokens, hidden_size = 6, 2
    # CLS=99, middle tokens=[3,4], EOS=99 → mean pooling → [3, 4] → norm=5
    hidden_data = np.zeros((1, n_tokens, hidden_size), dtype=np.float32)
    hidden_data[0, 0, :] = 99.0
    hidden_data[0, 1:-1, :] = np.array([[3.0, 4.0]] * (n_tokens - 2))
    hidden_data[0, -1, :] = 99.0
    model_output = MagicMock()
    model_output.last_hidden_state = FakeTensor(hidden_data)

    mock = _make_transformers_mod(n_tokens=n_tokens, hidden_size=hidden_size)
    model, tokenizer = _wire(loaded_backend, mock)
    model.return_value = model_output

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(pooling="mean", normalize=False))

    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# predict() — tokenizer + multiple sequences
# ---------------------------------------------------------------------------


def test_predict_tokenizer_called_per_sequence(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    """Tokenizer is called exactly once per sequence."""
    mock = _make_transformers_mod()
    model, tokenizer = _wire(loaded_backend, mock)
    seqs = ["ATCG", "GCTA", "TTAA"]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(_make_request(sequences=seqs))

    assert tokenizer.call_count == 3
    tokenizer.assert_has_calls(
        [call(seq, return_tensors="pt", truncation=True) for seq in seqs]
    )


def test_predict_multiple_sequences_shape(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    """Multiple sequences → one embedding per sequence."""
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    seqs = ["ATCG", "GCTA", "TTAA", "CCGG"]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(sequences=seqs))

    assert len(resp.embeddings) == 4
    assert all(len(e) == _HIDDEN_DIM for e in resp.embeddings)
    assert resp.dim == _HIDDEN_DIM


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_independently(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    mock = _make_transformers_mod()
    model, tokenizer = _wire(loaded_backend, mock)

    reqs = [
        _make_request(sequences=["ATCG"]),
        _make_request(sequences=["GCTA"]),
    ]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, GenomicResponse) for r in responses)
    assert tokenizer.call_count == 2
