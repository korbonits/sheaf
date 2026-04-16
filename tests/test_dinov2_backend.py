"""Tests for DINOv2Backend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes correct model_name to from_pretrained
  - load() moves model to the specified device
  - predict() rejects text inputs (image-only backbone)
  - predict() returns EmbeddingResponse with correct shape
  - predict() cls pooling — CLS token (position 0) extracted
  - predict() mean pooling — mean of patch tokens extracted
  - predict() normalize=True — unit-norm output
  - predict() normalize=False — raw embeddings returned
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
# FakeTensor — extended for DINOv2's hidden-state indexing and mean-pooling
# ---------------------------------------------------------------------------


class FakeTensor:
    """Numpy-backed tensor-like; supports slicing, norm, mean, division."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.array(data, dtype=np.float32)

    # --- tensor ops ---
    def norm(self, dim: int = -1, keepdim: bool = False) -> FakeTensor:
        norms = np.linalg.norm(self._data, axis=dim, keepdims=keepdim)
        return FakeTensor(norms)

    def mean(self, dim: int = 0) -> FakeTensor:
        return FakeTensor(self._data.mean(axis=dim))

    def __truediv__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(self._data / other._data)

    def __getitem__(self, idx: object) -> FakeTensor:
        return FakeTensor(self._data[idx])  # type: ignore[index]

    # --- device / dtype ---
    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self

    # --- list conversion ---
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
# Helpers
# ---------------------------------------------------------------------------

_EMBED_DIM = 768  # dinov2-base
_N = 2  # batch size
_SEQ_LEN = 197  # 1 CLS + 196 patches (14×14 ViT-B)


def _make_transformers_mod(
    embed_dim: int = _EMBED_DIM,
    n: int = _N,
    seq_len: int = _SEQ_LEN,
) -> ModuleType:
    mod = ModuleType("transformers")

    hidden = FakeTensor(np.random.randn(n, seq_len, embed_dim).astype(np.float32))

    outputs = MagicMock()
    outputs.last_hidden_state = hidden

    model = MagicMock()
    model.return_value = outputs  # model(**inputs) → outputs
    model.to.return_value = model

    mock_inputs: dict[str, FakeTensor] = {
        "pixel_values": FakeTensor(np.zeros((n, 3, 224, 224), dtype=np.float32))
    }
    processor = MagicMock()
    processor.return_value = mock_inputs

    mod.AutoImageProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoImageProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]
    mod.AutoModel = MagicMock()  # type: ignore[attr-defined]
    mod.AutoModel.from_pretrained.return_value = model  # type: ignore[attr-defined]

    return mod


def _make_pil_mock() -> tuple[ModuleType, MagicMock]:
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    image_cls.open.return_value = mock_img
    pil_mod.Image = image_cls  # type: ignore[attr-defined]
    return pil_mod, image_cls


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.dinov2 import DINOv2Backend

    backend = DINOv2Backend(model_name="facebook/dinov2-base", device="cpu")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    """Swap backend internals with fresh mock instances from mod."""
    model = mod.AutoModel.from_pretrained.return_value
    processor = mod.AutoImageProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    import builtins

    from sheaf.backends.dinov2 import DINOv2Backend

    backend = DINOv2Backend()
    # PIL is imported before transformers in load(); inject a mock so the
    # PIL import succeeds and only the transformers import raises.
    pil_mod, _ = _make_pil_mock()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    mods_without["PIL"] = pil_mod

    # Capture the real __import__ BEFORE patching — calling __import__ inside
    # _raise after the patch is applied would re-enter the mock and recurse.
    _real_import = builtins.__import__

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[vision\\]"),
    ):
        backend.load()


def test_load_passes_correct_model_name(mock_transformers: ModuleType) -> None:
    from sheaf.backends.dinov2 import DINOv2Backend

    backend = DINOv2Backend(model_name="facebook/dinov2-large")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoImageProcessor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "facebook/dinov2-large"
    )
    mock_transformers.AutoModel.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "facebook/dinov2-large"
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.dinov2 import DINOv2Backend

    backend = DINOv2Backend(device="cuda")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoModel.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_text_input(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = EmbeddingRequest(model_name="dinov2", texts=["a dog"])
    with pytest.raises(ValueError, match="image"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — image embeddings
# ---------------------------------------------------------------------------


def test_predict_returns_embedding_response(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    img_b64 = base64.b64encode(b"fake_image").decode()
    req = EmbeddingRequest(model_name="dinov2", images_b64=[img_b64, img_b64])
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, EmbeddingResponse)
    assert resp.dim == _EMBED_DIM
    assert len(resp.embeddings) == _N
    assert len(resp.embeddings[0]) == _EMBED_DIM


def test_predict_cls_pooling(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """CLS token is position 0 of last_hidden_state."""
    img_b64 = base64.b64encode(b"img").decode()
    req = EmbeddingRequest(model_name="dinov2", images_b64=[img_b64], normalize=False)
    loaded_backend._pooling = "cls"
    mock = _make_transformers_mod(n=1)
    model, _ = _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    # Craft hidden state where CLS (pos 0) has a known value
    known = np.zeros((_EMBED_DIM,), dtype=np.float32)
    known[0] = 7.0
    hidden_data = np.zeros((1, _SEQ_LEN, _EMBED_DIM), dtype=np.float32)
    hidden_data[0, 0, :] = known
    hidden_data[0, 1:, :] = 99.0  # patch tokens — should be ignored

    outputs = MagicMock()
    outputs.last_hidden_state = FakeTensor(hidden_data)
    model.return_value = outputs

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.embeddings[0][0] == pytest.approx(7.0)


def test_predict_mean_pooling(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Mean pooling averages patch tokens (positions 1:), ignores CLS."""
    img_b64 = base64.b64encode(b"img").decode()
    req = EmbeddingRequest(model_name="dinov2", images_b64=[img_b64], normalize=False)
    loaded_backend._pooling = "mean"
    mock = _make_transformers_mod(n=1)
    model, _ = _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    # CLS = 99 (should be excluded), patches = 2.0 (constant → mean = 2.0)
    hidden_data = np.full((1, _SEQ_LEN, _EMBED_DIM), 2.0, dtype=np.float32)
    hidden_data[0, 0, :] = 99.0  # CLS — ignored by mean pooling

    outputs = MagicMock()
    outputs.last_hidden_state = FakeTensor(hidden_data)
    model.return_value = outputs

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.embeddings[0][0] == pytest.approx(2.0)


def test_predict_normalize_true(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    img_b64 = base64.b64encode(b"img").decode()
    req = EmbeddingRequest(
        model_name="dinov2", images_b64=[img_b64, img_b64], normalize=True
    )
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    for emb in resp.embeddings:
        norm = math.sqrt(sum(x**2 for x in emb))
        assert abs(norm - 1.0) < 1e-5


def test_predict_normalize_false(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    img_b64 = base64.b64encode(b"img").decode()
    req = EmbeddingRequest(model_name="dinov2", images_b64=[img_b64], normalize=False)
    loaded_backend._pooling = "cls"
    mock = _make_transformers_mod(n=1)
    model, _ = _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    # CLS token: [3, 4, 0, 0, ...] → norm = 5
    hidden_data = np.zeros((1, _SEQ_LEN, _EMBED_DIM), dtype=np.float32)
    hidden_data[0, 0, 0] = 3.0
    hidden_data[0, 0, 1] = 4.0
    outputs = MagicMock()
    outputs.last_hidden_state = FakeTensor(hidden_data)
    model.return_value = outputs

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    norm = math.sqrt(sum(x**2 for x in resp.embeddings[0]))
    assert abs(norm - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    img_b64 = base64.b64encode(b"img").decode()
    reqs = [
        EmbeddingRequest(model_name="dinov2", images_b64=[img_b64]),
        EmbeddingRequest(model_name="dinov2", images_b64=[img_b64]),
    ]
    mock = _make_transformers_mod(n=1)
    model, processor = _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, EmbeddingResponse) for r in responses)
    assert model.call_count == 2
