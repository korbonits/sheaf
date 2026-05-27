"""Tests for ESMCBackend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() raises NotImplementedError for Forge-only model IDs
  - load() passes model_name through to from_pretrained
  - load() moves model to specified device when device_map is None
  - load() skips .to(device) when device_map is set
  - predict() rejects non-ProteinLanguageRequest inputs
  - predict() returns logits when return_logits=True
  - predict() omits logits when return_logits=False
  - predict() returns per-token embeddings when return_embeddings=True
  - predict() returns all-layer hidden_states when output_hidden_states=True
  - predict() slices outputs back to ragged per-sequence lengths via attention mask
  - predict() reports vocab_size and hidden_dim
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.protein_language import (
    ProteinLanguageRequest,
    ProteinLanguageResponse,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 5120  # ESMC 6B
_VOCAB = 33  # standard ESM amino-acid vocab (residues + special tokens)
_N_SEQ = 2
_SEQ_LEN = 8  # padded length; per-sequence real lengths = [6, 5]
_REAL_LENS = [6, 5]


# ---------------------------------------------------------------------------
# FakeTensor — numpy-backed; supports every op used in ESMCBackend._run()
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: list | np.ndarray, dtype: Any = np.float32) -> None:
        self._data = np.asarray(data, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(self._data[key])  # type: ignore[index]

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

    def to(self, _device: str) -> FakeTensor:
        return self


# ---------------------------------------------------------------------------
# Fake torch module
# ---------------------------------------------------------------------------


class _InferenceMode:
    def __enter__(self) -> _InferenceMode:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _make_torch_mod() -> ModuleType:
    mod = ModuleType("torch")
    mod.inference_mode = _InferenceMode  # type: ignore[attr-defined]
    return mod


_torch_mod = _make_torch_mod()


# ---------------------------------------------------------------------------
# Fake tokenizer output — dict-like with .items() and key access
# ---------------------------------------------------------------------------


def _make_attention_mask(real_lens: list[int], padded_len: int) -> np.ndarray:
    mask = np.zeros((len(real_lens), padded_len), dtype=np.int64)
    for i, n in enumerate(real_lens):
        mask[i, :n] = 1
    return mask


class _FakeTokenizerOutput:
    def __init__(
        self,
        real_lens: list[int] = _REAL_LENS,
        padded_len: int = _SEQ_LEN,
    ) -> None:
        self._d = {
            "input_ids": FakeTensor(
                np.zeros((len(real_lens), padded_len), dtype=np.int64),
                dtype=np.int64,
            ),
            "attention_mask": FakeTensor(
                _make_attention_mask(real_lens, padded_len),
                dtype=np.int64,
            ),
        }

    def items(self):  # type: ignore[no-untyped-def]
        return self._d.items()

    def __getitem__(self, key: str) -> FakeTensor:
        return self._d[key]


# ---------------------------------------------------------------------------
# Fake model output factory
# ---------------------------------------------------------------------------


def _make_model_output(
    n: int = _N_SEQ,
    seq_len: int = _SEQ_LEN,
    hidden_dim: int = _HIDDEN_DIM,
    vocab: int = _VOCAB,
    with_hidden_states: bool = False,
    n_layers: int = 4,
) -> MagicMock:
    logits = FakeTensor(np.full((n, seq_len, vocab), 0.5, dtype=np.float32))
    last_hidden = FakeTensor(np.full((n, seq_len, hidden_dim), 1.0, dtype=np.float32))
    out = MagicMock()
    out.logits = logits
    out.last_hidden_state = last_hidden
    if with_hidden_states:
        # Each layer is a (n, seq_len, hidden_dim) tensor; last is the same
        # as last_hidden_state.
        out.hidden_states = tuple(
            FakeTensor(np.full((n, seq_len, hidden_dim), float(i), dtype=np.float32))
            for i in range(n_layers)
        )
    else:
        out.hidden_states = None
    return out


# ---------------------------------------------------------------------------
# Fake transformers module factory
# ---------------------------------------------------------------------------


def _make_transformers_mod(
    with_hidden_states: bool = False,
) -> tuple[ModuleType, MagicMock, MagicMock]:
    model_output = _make_model_output(with_hidden_states=with_hidden_states)
    model = MagicMock()
    model.return_value = model_output
    model.to.return_value = model
    model.eval.return_value = None
    model.device = "cpu"

    tokenizer = MagicMock()
    tokenizer.return_value = _FakeTokenizerOutput()

    mod = ModuleType("transformers")
    mod.AutoModelForMaskedLM = MagicMock()  # type: ignore[attr-defined]
    mod.AutoModelForMaskedLM.from_pretrained.return_value = model  # type: ignore[attr-defined]
    mod.AutoTokenizer = MagicMock()  # type: ignore[attr-defined]
    mod.AutoTokenizer.from_pretrained.return_value = tokenizer  # type: ignore[attr-defined]
    return mod, model, tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()[0]


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(model_name="Biohub/ESMC-6B", device="cpu")
    with patch.dict(
        sys.modules, {"transformers": mock_transformers, "torch": _torch_mod}
    ):
        backend.load()
    return backend


def _wire_output(backend, *, with_hidden_states: bool = False) -> MagicMock:
    """Rebind the backend's stored model/tokenizer to fresh mocks."""
    mod, model, tokenizer = _make_transformers_mod(
        with_hidden_states=with_hidden_states
    )
    backend._model = model
    backend._tokenizer = tokenizer
    return model


def _make_request(
    sequences: list[str] | None = None,
    return_logits: bool = True,
    return_embeddings: bool = False,
    output_hidden_states: bool = False,
) -> ProteinLanguageRequest:
    if sequences is None:
        sequences = ["MKTII", "ACDE"]  # 5 and 4 residues
    return ProteinLanguageRequest(
        model_name="esmc",
        sequences=sequences,
        return_logits=return_logits,
        return_embeddings=return_embeddings,
        output_hidden_states=output_hidden_states,
    )


# ---------------------------------------------------------------------------
# load() — error cases
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[protein\\]"),
    ):
        backend.load()


def test_load_rejects_forge_only_model() -> None:
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(model_name="esmc-300m-2024-12")
    with pytest.raises(NotImplementedError, match="Forge"):
        backend.load()


def test_load_rejects_other_forge_only_model() -> None:
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(model_name="esmc-600m-2024-12")
    with pytest.raises(NotImplementedError, match="Forge"):
        backend.load()


# ---------------------------------------------------------------------------
# load() — happy path
# ---------------------------------------------------------------------------


def test_load_passes_model_name(mock_transformers: ModuleType) -> None:
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(model_name="Biohub/ESMC-6B")
    with patch.dict(
        sys.modules, {"transformers": mock_transformers, "torch": _torch_mod}
    ):
        backend.load()

    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "Biohub/ESMC-6B"
    )
    mock_transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "Biohub/ESMC-6B"
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(device="cuda")
    with patch.dict(
        sys.modules, {"transformers": mock_transformers, "torch": _torch_mod}
    ):
        backend.load()

    mock_transformers.AutoModelForMaskedLM.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


def test_load_device_map_skips_to_call(mock_transformers: ModuleType) -> None:
    """When device_map is set, .to(device) must not be called."""
    from sheaf.backends.esmc import ESMCBackend

    backend = ESMCBackend(device="cuda", device_map="auto")
    with patch.dict(
        sys.modules, {"transformers": mock_transformers, "torch": _torch_mod}
    ):
        backend.load()

    model = mock_transformers.AutoModelForMaskedLM.from_pretrained.return_value  # type: ignore[attr-defined]
    model.to.assert_not_called()
    mock_transformers.AutoModelForMaskedLM.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "Biohub/ESMC-6B", device_map="auto"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.molecular import MolecularRequest

    req = MolecularRequest(model_name="x", sequences=["MKT"])
    with pytest.raises(TypeError, match="ProteinLanguageRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure and slicing
# ---------------------------------------------------------------------------


def test_predict_returns_protein_language_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _wire_output(loaded_backend)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert isinstance(resp, ProteinLanguageResponse)
    assert resp.seq_lens == _REAL_LENS
    assert resp.vocab_size == _VOCAB
    assert resp.logits is not None
    assert resp.embeddings is None
    assert resp.hidden_states is None


def test_predict_slices_logits_to_real_lens(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """logits[i] must have length == seq_lens[i], not the padded length."""
    _wire_output(loaded_backend)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert resp.logits is not None
    assert [len(li) for li in resp.logits] == _REAL_LENS
    assert all(len(tok) == _VOCAB for li in resp.logits for tok in li)


def test_predict_skips_logits_when_disabled(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _wire_output(loaded_backend)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(
            _make_request(return_logits=False, return_embeddings=True)
        )

    assert resp.logits is None
    assert resp.vocab_size is None
    assert resp.embeddings is not None
    assert [len(ei) for ei in resp.embeddings] == _REAL_LENS
    assert resp.hidden_dim == _HIDDEN_DIM


def test_predict_returns_per_token_embeddings(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _wire_output(loaded_backend)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(return_embeddings=True))

    assert resp.embeddings is not None
    assert [len(ei) for ei in resp.embeddings] == _REAL_LENS
    assert resp.hidden_dim == _HIDDEN_DIM


def test_predict_output_hidden_states_returns_all_layers(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _wire_output(loaded_backend, with_hidden_states=True)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(
            _make_request(output_hidden_states=True, return_logits=False)
        )

    assert resp.hidden_states is not None
    assert len(resp.hidden_states) == 4  # n_layers from _make_model_output
    # Per-layer: list of per-sequence lists sliced to real lengths.
    for layer in resp.hidden_states:
        assert [len(ei) for ei in layer] == _REAL_LENS
    # And embeddings (last layer) is populated.
    assert resp.embeddings is not None
    assert [len(ei) for ei in resp.embeddings] == _REAL_LENS


# ---------------------------------------------------------------------------
# predict() — tokenizer call shape
# ---------------------------------------------------------------------------


def test_predict_tokenizes_full_batch_with_padding(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    model = _wire_output(loaded_backend)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(_make_request(sequences=["MKTII", "ACDE"]))

    # Tokenizer called with the full sequences list, padding=True
    tokenizer = loaded_backend._tokenizer
    args, kwargs = tokenizer.call_args
    assert args[0] == ["MKTII", "ACDE"]
    assert kwargs["padding"] is True
    assert kwargs["return_tensors"] == "pt"
    # One forward pass per batch
    assert model.call_count == 1


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_each_request(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    model = _wire_output(loaded_backend)
    reqs = [_make_request(), _make_request(sequences=["AKDQE", "MKTL"])]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, ProteinLanguageResponse) for r in responses)
    assert model.call_count == 2
