"""Tests for DETRBackend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes correct model_name to from_pretrained
  - load() moves model to the specified device
  - predict() rejects non-DetectionRequest inputs
  - predict() returns DetectionResponse with correct structure
  - predict() box coordinates match post_process output
  - predict() label strings resolved via model.config.id2label
  - predict() threshold forwarded to post_process_object_detection
  - predict() empty detections — all lists empty
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.detection import DetectionRequest, DetectionResponse

# ---------------------------------------------------------------------------
# FakeTensor — cpu() + tolist() for boxes / scores
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: list | np.ndarray) -> None:
        self._data = data if isinstance(data, list) else data.tolist()

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list:  # type: ignore[override]
        return self._data

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMG_B64 = base64.b64encode(b"fake_image_bytes").decode()
_IMG_SIZE = (640, 480)  # (W, H) — PIL convention

_BOXES = [[10.0, 20.0, 100.0, 200.0], [50.0, 60.0, 150.0, 160.0]]
_SCORES = [0.92, 0.85]
_LABEL_IDS = [0, 1]
_ID2LABEL = {0: "cat", 1: "dog"}


def _make_label_ids(ids: list[int]) -> list[MagicMock]:
    """Return list of MagicMocks where .item() returns the integer id."""
    mocks = []
    for i in ids:
        m = MagicMock()
        m.item.return_value = i
        mocks.append(m)
    return mocks


def _make_post_process_result(
    boxes: list[list[float]] = _BOXES,
    scores: list[float] = _SCORES,
    label_ids: list[int] = _LABEL_IDS,
) -> dict:
    return {
        "boxes": FakeTensor(boxes),
        "scores": FakeTensor(scores),
        "labels": _make_label_ids(label_ids),
    }


def _make_transformers_mod(
    boxes: list[list[float]] = _BOXES,
    scores: list[float] = _SCORES,
    label_ids: list[int] = _LABEL_IDS,
    id2label: dict[int, str] = _ID2LABEL,
) -> ModuleType:
    outputs = MagicMock()

    result = _make_post_process_result(boxes, scores, label_ids)
    processor = MagicMock()
    processor.return_value = {"pixel_values": FakeTensor([[0.0]])}
    processor.post_process_object_detection.return_value = [result]

    model = MagicMock()
    model.return_value = outputs
    model.to.return_value = model
    model.config.id2label = id2label

    mod = ModuleType("transformers")
    mod.AutoImageProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoImageProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]
    mod.AutoModelForObjectDetection = MagicMock()  # type: ignore[attr-defined]
    mod.AutoModelForObjectDetection.from_pretrained.return_value = model  # type: ignore[attr-defined]
    return mod


def _make_pil_mock(
    size: tuple[int, int] = _IMG_SIZE,
) -> tuple[ModuleType, MagicMock]:
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.size = size  # (W, H) — PIL convention
    image_cls.open.return_value = mock_img
    pil_mod.Image = image_cls  # type: ignore[attr-defined]
    return pil_mod, image_cls


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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.detr import DETRBackend

    backend = DETRBackend(model_name="facebook/detr-resnet-50", device="cpu")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    model = mod.AutoModelForObjectDetection.from_pretrained.return_value
    processor = mod.AutoImageProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    import builtins

    from sheaf.backends.detr import DETRBackend

    backend = DETRBackend()
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
    from sheaf.backends.detr import DETRBackend

    backend = DETRBackend(model_name="PekingU/rtdetr_r50vd")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoImageProcessor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "PekingU/rtdetr_r50vd"
    )
    mock_transformers.AutoModelForObjectDetection.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "PekingU/rtdetr_r50vd"
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.detr import DETRBackend

    backend = DETRBackend(device="cuda")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoModelForObjectDetection.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="DetectionRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_detection_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, processor = _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[0].Image

    req = DetectionRequest(model_name="detr", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, DetectionResponse)
    assert len(resp.boxes) == len(_BOXES)
    assert len(resp.scores) == len(_SCORES)
    assert len(resp.labels) == len(_LABEL_IDS)
    assert resp.width == _IMG_SIZE[0]
    assert resp.height == _IMG_SIZE[1]


def test_predict_box_coordinates(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[0].Image

    req = DetectionRequest(model_name="detr", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.boxes == _BOXES


def test_predict_label_strings(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[0].Image

    req = DetectionRequest(model_name="detr", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.labels == ["cat", "dog"]


def test_predict_threshold_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """threshold is passed through to post_process_object_detection."""
    mock = _make_transformers_mod()
    model, processor = _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[0].Image

    req = DetectionRequest(model_name="detr", image_b64=_IMG_B64, threshold=0.7)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(req)

    _, kwargs = processor.post_process_object_detection.call_args
    assert kwargs["threshold"] == pytest.approx(0.7)


def test_predict_target_sizes_use_image_dimensions(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """target_sizes passed as [(height, width)] using the PIL image size."""
    img_w, img_h = 800, 600
    mock = _make_transformers_mod()
    model, processor = _wire(loaded_backend, mock)
    pil_mod, _ = _make_pil_mock(size=(img_w, img_h))
    loaded_backend._Image = pil_mod.Image

    req = DetectionRequest(model_name="detr", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    _, kwargs = processor.post_process_object_detection.call_args
    assert kwargs["target_sizes"] == [(img_h, img_w)]
    assert resp.width == img_w
    assert resp.height == img_h


def test_predict_empty_detections(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """No detections above threshold → all lists empty."""
    mock = _make_transformers_mod(boxes=[], scores=[], label_ids=[])
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[0].Image

    req = DetectionRequest(model_name="detr", image_b64=_IMG_B64, threshold=0.99)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.boxes == []
    assert resp.scores == []
    assert resp.labels == []


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[0].Image

    reqs = [
        DetectionRequest(model_name="detr", image_b64=_IMG_B64),
        DetectionRequest(model_name="detr", image_b64=_IMG_B64),
    ]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, DetectionResponse) for r in responses)
    assert model.call_count == 2
