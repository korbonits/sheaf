"""Tests for RAFTBackend — fully mocked, no torchvision/torch/PIL required.

Covers:
  - __init__() raises ValueError on invalid model_name
  - load() raises ImportError when torchvision is absent
  - load() uses raft_large weights + model for "raft_large"
  - load() uses raft_small weights + model for "raft_small"
  - load() moves model to the specified device
  - model_type returns ModelType.OPTICAL_FLOW
  - predict() rejects non-OpticalFlowRequest inputs
  - predict() returns OpticalFlowResponse with correct structure
  - predict() flow_b64 decodes to (H, W, 2) float32 array
  - predict() output dimensions match input frame size
  - predict() crops padding from RAFT output correctly
  - predict() applies transforms before model call
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.optical_flow import OpticalFlowRequest, OpticalFlowResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_H, _W = 480, 640
_FRAME_B64 = base64.b64encode(b"fake_frame_bytes").decode()
_REQ = OpticalFlowRequest(
    model_name="raft",
    frame1_b64=_FRAME_B64,
    frame2_b64=_FRAME_B64,
)

# ---------------------------------------------------------------------------
# Fake image — supports np.array() without PIL installed
# ---------------------------------------------------------------------------


class _FakeImage:
    def __init__(self, w: int = _W, h: int = _H) -> None:
        self._w, self._h = w, h

    @property
    def size(self) -> tuple[int, int]:  # PIL convention: (W, H)
        return (self._w, self._h)

    def convert(self, mode: str) -> _FakeImage:
        return self

    def __array__(self, dtype=None) -> np.ndarray:  # type: ignore[override]
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fake tensor — supports the chain .permute().unsqueeze().to()
# ---------------------------------------------------------------------------


class _FakeTensor:
    def permute(self, *args: int) -> _FakeTensor:
        return self

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return self

    def to(self, device: str) -> _FakeTensor:
        return self


# ---------------------------------------------------------------------------
# Fake torch
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_) -> None:  # type: ignore[no-untyped-def]
        pass


def _make_torch_mod(flow_np: np.ndarray | None = None) -> ModuleType:
    """Build a minimal fake torch module producing a fixed flow array."""
    _flow = flow_np if flow_np is not None else np.zeros((_H, _W, 2), dtype=np.float32)
    # Reconstruct what the backend expects as (2, H, W) before transposing.
    flow_chw = _flow.transpose(2, 0, 1)  # (2, H, W)

    # flow_predictions[-1][0].cpu().numpy() → flow_chw
    inner = MagicMock()
    inner.cpu.return_value.numpy.return_value = flow_chw

    flow_batch = MagicMock()
    flow_batch.__getitem__ = MagicMock(return_value=inner)

    flow_preds = MagicMock()
    flow_preds.__getitem__ = MagicMock(return_value=flow_batch)

    mod = ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]

    fake_t = _FakeTensor()
    mod.from_numpy = MagicMock(return_value=fake_t)  # type: ignore[attr-defined]
    return mod, flow_preds  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# torchvision mock factory
# ---------------------------------------------------------------------------


def _make_torchvision_mod(
    model_name: str = "raft_large",
) -> ModuleType:
    mod = ModuleType("torchvision.models.optical_flow")

    for name in ("raft_large", "raft_small"):
        weights_cls = MagicMock()
        weights_cls.DEFAULT = MagicMock()
        weights_cls.DEFAULT.transforms.return_value = MagicMock(
            side_effect=lambda a, b: (a, b)
        )

        model_instance = MagicMock()
        model_instance.to.return_value = model_instance

        model_fn = MagicMock(return_value=model_instance)

        setattr(
            mod, f"Raft_{'Large' if 'large' in name else 'Small'}_Weights", weights_cls
        )
        setattr(mod, name, model_fn)

    return mod


# ---------------------------------------------------------------------------
# PIL mock factory
# ---------------------------------------------------------------------------


def _make_pil_mock(w: int = _W, h: int = _H) -> tuple[ModuleType, MagicMock]:
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    image_cls.open.return_value = _FakeImage(w, h)
    pil_mod.Image = image_cls  # type: ignore[attr-defined]
    return pil_mod, image_cls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tv() -> ModuleType:
    return _make_torchvision_mod()


@pytest.fixture
def torch_mod_and_preds():  # type: ignore[no-untyped-def]
    torch_mod, flow_preds = _make_torch_mod()
    return torch_mod, flow_preds


@pytest.fixture
def loaded_backend(mock_tv: ModuleType, torch_mod_and_preds):  # type: ignore[no-untyped-def]
    from sheaf.backends.raft import RAFTBackend

    backend = RAFTBackend(model_name="raft_large", device="cpu")
    pil_mod, _ = _make_pil_mock()

    with patch.dict(
        sys.modules,
        {
            "torchvision.models.optical_flow": mock_tv,
            "PIL": pil_mod,
        },
    ):
        backend.load()

    torch_mod, flow_preds = torch_mod_and_preds
    model_instance = mock_tv.raft_large.return_value
    model_instance.return_value = flow_preds
    backend._model = model_instance
    backend._Image = _make_pil_mock()[1]
    backend._transforms = lambda a, b: (a, b)
    return backend, torch_mod, flow_preds


# ---------------------------------------------------------------------------
# __init__() validation
# ---------------------------------------------------------------------------


def test_init_rejects_invalid_model_name() -> None:
    from sheaf.backends.raft import RAFTBackend

    with pytest.raises(ValueError, match="raft_large"):
        RAFTBackend(model_name="invalid_model")


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_torchvision() -> None:
    import builtins

    from sheaf.backends.raft import RAFTBackend

    backend = RAFTBackend()
    pil_mod, _ = _make_pil_mock()
    mods_without = {k: v for k, v in sys.modules.items() if "torchvision" not in k}
    mods_without["PIL"] = pil_mod
    _real_import = builtins.__import__

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if "torchvision" in name:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[optical-flow\\]"),
    ):
        backend.load()


def test_load_uses_raft_large(mock_tv: ModuleType) -> None:
    from sheaf.backends.raft import RAFTBackend

    backend = RAFTBackend(model_name="raft_large")
    pil_mod, _ = _make_pil_mock()

    with patch.dict(
        sys.modules, {"torchvision.models.optical_flow": mock_tv, "PIL": pil_mod}
    ):
        backend.load()

    mock_tv.raft_large.assert_called_once()  # type: ignore[attr-defined]
    mock_tv.raft_small.assert_not_called()  # type: ignore[attr-defined]


def test_load_uses_raft_small(mock_tv: ModuleType) -> None:
    from sheaf.backends.raft import RAFTBackend

    backend = RAFTBackend(model_name="raft_small")
    pil_mod, _ = _make_pil_mock()

    with patch.dict(
        sys.modules, {"torchvision.models.optical_flow": mock_tv, "PIL": pil_mod}
    ):
        backend.load()

    mock_tv.raft_small.assert_called_once()  # type: ignore[attr-defined]
    mock_tv.raft_large.assert_not_called()  # type: ignore[attr-defined]


def test_load_moves_model_to_device(mock_tv: ModuleType) -> None:
    from sheaf.backends.raft import RAFTBackend

    backend = RAFTBackend(device="cuda")
    pil_mod, _ = _make_pil_mock()

    with patch.dict(
        sys.modules, {"torchvision.models.optical_flow": mock_tv, "PIL": pil_mod}
    ):
        backend.load()

    mock_tv.raft_large.return_value.to.assert_called_once_with("cuda")  # type: ignore[attr-defined]


def test_model_type() -> None:
    from sheaf.api.base import ModelType
    from sheaf.backends.raft import RAFTBackend

    assert RAFTBackend().model_type == ModelType.OPTICAL_FLOW


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    backend, torch_mod, _ = loaded_backend
    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="OpticalFlowRequest"):
        backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_optical_flow_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, _ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod}):
        resp = backend.predict(_REQ)
    assert isinstance(resp, OpticalFlowResponse)


def test_predict_flow_b64_decodes_to_correct_shape(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, _ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod}):
        resp = backend.predict(_REQ)

    flow = np.frombuffer(base64.b64decode(resp.flow_b64), dtype=np.float32).reshape(
        resp.height, resp.width, 2
    )
    assert flow.shape == (_H, _W, 2)


def test_predict_dimensions_match_input(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, _ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod}):
        resp = backend.predict(_REQ)
    assert resp.width == _W
    assert resp.height == _H


def test_predict_crops_padding_correctly() -> None:
    """When model output is larger than input (padded), flow is cropped to orig size."""
    from sheaf.backends.raft import RAFTBackend

    orig_h, orig_w = 100, 120
    pad_h, pad_w = 104, 128  # padded to multiples of 8

    flow_padded_chw = np.ones((2, pad_h, pad_w), dtype=np.float32) * 5.0

    inner = MagicMock()
    inner.cpu.return_value.numpy.return_value = flow_padded_chw
    flow_batch = MagicMock()
    flow_batch.__getitem__ = MagicMock(return_value=inner)
    flow_preds = MagicMock()
    flow_preds.__getitem__ = MagicMock(return_value=flow_batch)

    class _NoGrad2:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    torch_mod = ModuleType("torch")
    torch_mod.no_grad = _NoGrad2  # type: ignore[attr-defined]
    torch_mod.from_numpy = MagicMock(return_value=_FakeTensor())  # type: ignore[attr-defined]

    backend = RAFTBackend(model_name="raft_large", device="cpu")
    backend._model = MagicMock(return_value=flow_preds)
    backend._model.to = MagicMock(return_value=backend._model)
    backend._transforms = lambda a, b: (a, b)
    backend._Image = _make_pil_mock(w=orig_w, h=orig_h)[1]

    req = OpticalFlowRequest(
        model_name="raft", frame1_b64=_FRAME_B64, frame2_b64=_FRAME_B64
    )
    with patch.dict(sys.modules, {"torch": torch_mod}):
        resp = backend.predict(req)

    assert resp.width == orig_w
    assert resp.height == orig_h
    flow = np.frombuffer(base64.b64decode(resp.flow_b64), dtype=np.float32).reshape(
        resp.height, resp.width, 2
    )
    assert flow.shape == (orig_h, orig_w, 2)
    assert np.allclose(flow, 5.0)


def test_predict_transforms_applied(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Verify the transforms callable is invoked during predict."""
    backend, torch_mod, _ = loaded_backend
    transforms_mock = MagicMock(side_effect=lambda a, b: (a, b))
    backend._transforms = transforms_mock

    with patch.dict(sys.modules, {"torch": torch_mod}):
        backend.predict(_REQ)

    transforms_mock.assert_called_once()


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, _ = loaded_backend
    reqs = [_REQ, _REQ]
    with patch.dict(sys.modules, {"torch": torch_mod}):
        responses = backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, OpticalFlowResponse) for r in responses)
    assert backend._model.call_count == 2
