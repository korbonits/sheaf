"""Tests for DepthAnythingBackend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes correct model_name to from_pretrained
  - load() moves model to the specified device
  - predict() rejects non-DepthRequest inputs
  - predict() returns DepthResponse with correct metadata
  - predict() normalize=True — depth values in [0, 1]
  - predict() normalize=False — raw depth values returned; bounds preserved
  - predict() depth encoding round-trip — float32 bytes decode to original array
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.depth import DepthRequest, DepthResponse

# ---------------------------------------------------------------------------
# FakeTensor — numpy-backed; supports squeeze / cpu / numpy for depth output
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.array(data, dtype=np.float32)

    def squeeze(self) -> FakeTensor:
        return FakeTensor(np.squeeze(self._data))

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._data

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self

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

_H, _W = 32, 32
_IMG_B64 = base64.b64encode(b"fake_image_bytes").decode()


# ---------------------------------------------------------------------------
# Fake transformers module factory
# ---------------------------------------------------------------------------


def _make_transformers_mod(h: int = _H, w: int = _W) -> ModuleType:
    depth_data = np.random.rand(1, h, w).astype(np.float32) * 10.0  # raw depth

    outputs = MagicMock()
    outputs.predicted_depth = FakeTensor(depth_data)

    model = MagicMock()
    model.return_value = outputs
    model.to.return_value = model

    processor = MagicMock()
    processor.return_value = {"pixel_values": FakeTensor(np.zeros((1, 3, 518, 518)))}

    mod = ModuleType("transformers")
    mod.AutoImageProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoImageProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]
    mod.AutoModelForDepthEstimation = MagicMock()  # type: ignore[attr-defined]
    mod.AutoModelForDepthEstimation.from_pretrained.return_value = model  # type: ignore[attr-defined]
    return mod


def _make_pil_mock() -> tuple[ModuleType, MagicMock]:
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    image_cls.open.return_value = mock_img
    pil_mod.Image = image_cls  # type: ignore[attr-defined]
    return pil_mod, image_cls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.depth_anything import DepthAnythingBackend

    backend = DepthAnythingBackend(
        model_name="depth-anything/Depth-Anything-V2-Small-hf", device="cpu"
    )
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    model = mod.AutoModelForDepthEstimation.from_pretrained.return_value
    processor = mod.AutoImageProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    import builtins

    from sheaf.backends.depth_anything import DepthAnythingBackend

    backend = DepthAnythingBackend()
    pil_mod, _ = _make_pil_mock()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    mods_without["PIL"] = pil_mod
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
    from sheaf.backends.depth_anything import DepthAnythingBackend

    large = "depth-anything/Depth-Anything-V2-Large-hf"
    backend = DepthAnythingBackend(model_name=large)
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoImageProcessor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        large
    )
    mock_transformers.AutoModelForDepthEstimation.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        large
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.depth_anything import DepthAnythingBackend

    backend = DepthAnythingBackend(device="cuda")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoModelForDepthEstimation.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="DepthRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response shape and metadata
# ---------------------------------------------------------------------------


def test_predict_returns_depth_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    req = DepthRequest(model_name="depth-anything", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, DepthResponse)
    assert resp.height == _H
    assert resp.width == _W
    assert resp.min_depth <= resp.max_depth


# ---------------------------------------------------------------------------
# predict() — normalization
# ---------------------------------------------------------------------------


def test_predict_normalize_true(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """normalize=True → depth values in [0, 1]."""
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    req = DepthRequest(model_name="depth-anything", image_b64=_IMG_B64, normalize=True)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    depth = np.frombuffer(base64.b64decode(resp.depth_b64), dtype=np.float32).reshape(
        resp.height, resp.width
    )
    assert float(depth.min()) >= -1e-6
    assert float(depth.max()) <= 1.0 + 1e-6


def test_predict_normalize_false(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """normalize=False → raw values; min/max bounds match encoded map."""
    known = np.array([[1.0, 5.0], [3.0, 10.0]], dtype=np.float32)
    outputs = MagicMock()
    outputs.predicted_depth = FakeTensor(known[np.newaxis])  # (1, 2, 2)

    model = loaded_backend._model
    model.return_value = outputs

    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()
    # Ensure processor returns a minimal dict
    loaded_backend._processor.return_value = {
        "pixel_values": FakeTensor(np.zeros((1, 3, 4, 4)))
    }

    req = DepthRequest(model_name="depth-anything", image_b64=_IMG_B64, normalize=False)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.min_depth == pytest.approx(1.0)
    assert resp.max_depth == pytest.approx(10.0)

    depth = np.frombuffer(base64.b64decode(resp.depth_b64), dtype=np.float32).reshape(
        resp.height, resp.width
    )
    np.testing.assert_allclose(depth, known, rtol=1e-5)


# ---------------------------------------------------------------------------
# predict() — encoding round-trip
# ---------------------------------------------------------------------------


def test_predict_depth_encoding_round_trip(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """depth_b64 decodes back to the original float32 depth map."""
    known = np.linspace(0.5, 9.5, _H * _W, dtype=np.float32).reshape(_H, _W)
    outputs = MagicMock()
    outputs.predicted_depth = FakeTensor(known[np.newaxis])

    loaded_backend._model.return_value = outputs
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()
    loaded_backend._processor.return_value = {
        "pixel_values": FakeTensor(np.zeros((1, 3, 518, 518)))
    }

    req = DepthRequest(model_name="depth-anything", image_b64=_IMG_B64, normalize=False)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    decoded = np.frombuffer(base64.b64decode(resp.depth_b64), dtype=np.float32).reshape(
        resp.height, resp.width
    )
    np.testing.assert_allclose(decoded, known, rtol=1e-6)


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)
    loaded_backend._Image = MagicMock()
    loaded_backend._Image.open.return_value.convert.return_value = MagicMock()

    reqs = [
        DepthRequest(model_name="depth-anything", image_b64=_IMG_B64),
        DepthRequest(model_name="depth-anything", image_b64=_IMG_B64),
    ]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, DepthResponse) for r in responses)
    assert model.call_count == 2
