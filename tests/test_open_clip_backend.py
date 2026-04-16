"""Tests for OpenCLIPBackend — fully mocked, no open-clip-torch or torch required.

Covers:
  - load() raises ImportError when open-clip-torch is absent
  - load() passes correct model_name and pretrained to create_model_and_transforms
  - EmbeddingRequest validation: rejects neither-input and both-inputs
  - predict() text embedding — returns EmbeddingResponse with correct shape
  - predict() normalize=True — output embeddings have unit norm
  - predict() normalize=False — raw embeddings returned unchanged
  - predict() dim field matches embedding dimensionality
  - predict() image embedding — images decoded, preprocess called per image
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse

# ---------------------------------------------------------------------------
# FakeTensor — minimal tensor-like for testing without torch installed
# ---------------------------------------------------------------------------


class FakeTensor:
    """Numpy-backed object that quacks like a torch.Tensor for our backend."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.array(data, dtype=np.float32)

    def norm(self, dim: int = -1, keepdim: bool = False) -> FakeTensor:
        norms = np.linalg.norm(self._data, axis=dim, keepdims=keepdim)
        return FakeTensor(norms)

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self._data / other._data)

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def tolist(self) -> list:  # type: ignore[override]
        return self._data.tolist()

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self

    @property
    def shape(self) -> tuple:
        return self._data.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fake torch module — provides no_grad() context and stack()
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_) -> None:  # type: ignore[no-untyped-def]
        pass


def _make_torch_mod() -> ModuleType:
    mod = ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]

    def _stack(tensors, dim: int = 0):  # type: ignore[no-untyped-def]
        return FakeTensor(np.stack([t._data for t in tensors], axis=dim))

    mod.stack = _stack  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 512
_N = 2  # batch size used in mock returns


def _make_open_clip_mod(embed_dim: int = _EMBED_DIM, n: int = _N) -> ModuleType:
    mod = ModuleType("open_clip")

    emb = FakeTensor(np.random.randn(n, embed_dim).astype(np.float32))

    model = MagicMock()
    model.encode_text.return_value = emb
    model.encode_image.return_value = emb

    tokenizer = MagicMock()
    # tokenizer(texts) → FakeTensor; backend checks hasattr(tokens, "to")
    tokenizer.return_value = FakeTensor(np.zeros((n, 77), dtype=np.float32))

    preprocess = MagicMock()
    preprocess.return_value = FakeTensor(np.zeros((3, 224, 224), dtype=np.float32))

    mod.create_model_and_transforms = MagicMock(  # type: ignore[attr-defined]
        return_value=(model, None, preprocess)
    )
    mod.get_tokenizer = MagicMock(return_value=tokenizer)  # type: ignore[attr-defined]

    return mod


def _make_pil_mock() -> tuple[ModuleType, MagicMock]:
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    image_cls.open.return_value = mock_img
    pil_mod.Image = image_cls  # type: ignore[attr-defined]
    return pil_mod, image_cls


_torch_mod = _make_torch_mod()


@pytest.fixture
def mock_open_clip() -> ModuleType:
    return _make_open_clip_mod()


@pytest.fixture
def loaded_backend(mock_open_clip: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.open_clip import OpenCLIPBackend

    backend = OpenCLIPBackend(model_name="ViT-B-32", pretrained="openai", device="cpu")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"open_clip": mock_open_clip, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Swap backend internals with fresh instances from mod."""
    model, _, preprocess = mod.create_model_and_transforms.return_value
    tokenizer = mod.get_tokenizer.return_value
    backend._model = model
    backend._preprocess = preprocess
    backend._tokenizer = tokenizer
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_open_clip() -> None:
    from sheaf.backends.open_clip import OpenCLIPBackend

    backend = OpenCLIPBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "open_clip" not in k}

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "open_clip":
            raise ModuleNotFoundError("No module named 'open_clip'")
        return __import__(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[vision\\]"),
    ):
        backend.load()


def test_load_passes_correct_args(mock_open_clip: ModuleType) -> None:
    from sheaf.backends.open_clip import OpenCLIPBackend

    backend = OpenCLIPBackend(
        model_name="ViT-L-14", pretrained="laion2b_s32b_b79k", device="cpu"
    )
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"open_clip": mock_open_clip, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_open_clip.create_model_and_transforms.assert_called_once_with(  # type: ignore[attr-defined]
        "ViT-L-14",
        pretrained="laion2b_s32b_b79k",
        device="cpu",
    )
    mock_open_clip.get_tokenizer.assert_called_once_with("ViT-L-14")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# EmbeddingRequest validation
# ---------------------------------------------------------------------------


def test_request_rejects_neither_input() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        EmbeddingRequest(model_name="open-clip")


def test_request_rejects_both_inputs() -> None:
    with pytest.raises(ValueError, match="not both"):
        EmbeddingRequest(
            model_name="open-clip",
            texts=["hello"],
            images_b64=[base64.b64encode(b"img").decode()],
        )


def test_request_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError, match="base64"):
        EmbeddingRequest(model_name="open-clip", images_b64=["not-valid-base64!!!"])


# ---------------------------------------------------------------------------
# predict() — text
# ---------------------------------------------------------------------------


def test_predict_text_returns_embedding_response(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    req = EmbeddingRequest(model_name="open-clip", texts=["a photo of a dog", "cat"])
    mock = _make_open_clip_mod()
    _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, EmbeddingResponse)
    assert resp.dim == _EMBED_DIM
    assert len(resp.embeddings) == _N
    assert len(resp.embeddings[0]) == _EMBED_DIM


def test_predict_text_normalize_true(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    req = EmbeddingRequest(model_name="open-clip", texts=["cat", "dog"], normalize=True)
    mock = _make_open_clip_mod()
    model, _, _ = _wire(loaded_backend, mock)
    # Use a known non-unit vector so we can verify normalization
    model.encode_text.return_value = FakeTensor(
        np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    )

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    for emb in resp.embeddings:
        norm = math.sqrt(sum(x**2 for x in emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_predict_text_normalize_false(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    req = EmbeddingRequest(model_name="open-clip", texts=["cat"], normalize=False)
    mock = _make_open_clip_mod()
    model, _, _ = _wire(loaded_backend, mock)
    model.encode_text.return_value = FakeTensor(
        np.array([[3.0, 4.0]], dtype=np.float32)
    )

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 5.0) < 1e-4  # 3-4-5, not normalized


def test_predict_text_dim_field(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    req = EmbeddingRequest(model_name="open-clip", texts=["hello"], normalize=False)
    mock = _make_open_clip_mod(embed_dim=768, n=1)
    _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.dim == 768


# ---------------------------------------------------------------------------
# predict() — image
# ---------------------------------------------------------------------------


def test_predict_image_returns_embedding_response(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    img_b64 = base64.b64encode(b"fake_image_bytes").decode()
    req = EmbeddingRequest(model_name="open-clip", images_b64=[img_b64, img_b64])
    mock = _make_open_clip_mod()
    _wire(loaded_backend, mock)
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value = mock_img

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, EmbeddingResponse)
    assert resp.dim == _EMBED_DIM
    assert len(resp.embeddings) == _N


def test_predict_image_preprocess_called_per_image(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    img_b64 = base64.b64encode(b"fake_image_bytes").decode()
    req = EmbeddingRequest(model_name="open-clip", images_b64=[img_b64, img_b64])
    mock = _make_open_clip_mod()
    model, preprocess, _ = _wire(loaded_backend, mock)
    model.encode_image.return_value = FakeTensor(
        np.random.randn(2, _EMBED_DIM).astype(np.float32)
    )
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value = mock_img

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(req)

    assert preprocess.call_count == 2


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    reqs = [
        EmbeddingRequest(model_name="open-clip", texts=["cat"]),
        EmbeddingRequest(model_name="open-clip", texts=["dog"]),
    ]
    mock = _make_open_clip_mod(n=1)
    model, _, _ = _wire(loaded_backend, mock)
    model.encode_text.return_value = FakeTensor(
        np.random.randn(1, _EMBED_DIM).astype(np.float32)
    )

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, EmbeddingResponse) for r in responses)
    assert model.encode_text.call_count == 2
