"""Tests for ViTPoseBackend — fully mocked, no transformers/torch/PIL required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes correct model_name to from_pretrained
  - load() moves model to the specified device
  - model_type returns ModelType.POSE
  - predict() rejects non-PoseRequest inputs
  - predict() returns PoseResponse with correct structure
  - predict() keypoints are [x, y, score] triples in pixel coords
  - predict() uses bboxes from request when provided
  - predict() defaults to full-image bbox when bboxes is None
  - predict() returns keypoint_names from model.config.id2label
  - predict() handles missing id2label gracefully (returns empty list)
  - predict() returns correct image dimensions (width, height)
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.pose import PoseRequest, PoseResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMG_B64 = base64.b64encode(b"fake_image_bytes").decode()
_IMG_SIZE = (640, 480)  # (W, H) — PIL convention

_COCO_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Two fake persons, 17 keypoints each
_KPT_XY_P0 = [[float(i * 10), float(i * 5)] for i in range(17)]
_KPT_SCORES_P0 = [0.9 - i * 0.01 for i in range(17)]
_KPT_XY_P1 = [[float(i * 8), float(i * 4)] for i in range(17)]
_KPT_SCORES_P1 = [0.8 - i * 0.01 for i in range(17)]


# ---------------------------------------------------------------------------
# Fake tensor helpers
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: list) -> None:
        self._data = data

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list:  # type: ignore[override]
        return self._data

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Fake torch
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
# Factory helpers
# ---------------------------------------------------------------------------


def _make_person_result(
    kpt_xy: list[list[float]],
    scores: list[float],
) -> dict:
    return {
        "keypoints": FakeTensor(kpt_xy),
        "scores": FakeTensor(scores),
    }


def _make_transformers_mod(
    persons: list[dict] | None = None,
    id2label: dict[int, str] | None = None,
) -> ModuleType:
    if persons is None:
        persons = [_make_person_result(_KPT_XY_P0, _KPT_SCORES_P0)]
    if id2label is None:
        id2label = {i: name for i, name in enumerate(_COCO_NAMES)}

    processor = MagicMock()
    processor.return_value = {"pixel_values": FakeTensor([[0.0]])}
    processor.post_process_pose_estimation.return_value = [persons]

    model = MagicMock()
    model.to.return_value = model
    model.config.id2label = id2label

    mod = ModuleType("transformers")
    mod.AutoProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]
    mod.VitPoseForPoseEstimation = MagicMock()  # type: ignore[attr-defined]
    mod.VitPoseForPoseEstimation.from_pretrained.return_value = model  # type: ignore[attr-defined]
    return mod


def _make_pil_mock(size: tuple[int, int] = _IMG_SIZE) -> tuple[ModuleType, MagicMock]:
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_img.size = size
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
    from sheaf.backends.vitpose import ViTPoseBackend

    backend = ViTPoseBackend(
        model_name="usyd-community/vitpose-base-simple", device="cpu"
    )
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    model = mod.VitPoseForPoseEstimation.from_pretrained.return_value
    processor = mod.AutoProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    import builtins

    from sheaf.backends.vitpose import ViTPoseBackend

    backend = ViTPoseBackend()
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
        pytest.raises(ImportError, match="sheaf-serve\\[pose\\]"),
    ):
        backend.load()


def test_load_passes_correct_model_name(mock_transformers: ModuleType) -> None:
    from sheaf.backends.vitpose import ViTPoseBackend

    backend = ViTPoseBackend(model_name="usyd-community/vitpose-plus-base")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.AutoProcessor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "usyd-community/vitpose-plus-base"
    )
    mock_transformers.VitPoseForPoseEstimation.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "usyd-community/vitpose-plus-base"
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.vitpose import ViTPoseBackend

    backend = ViTPoseBackend(device="cuda")
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {"transformers": mock_transformers, "PIL": pil_mod, "torch": _torch_mod},
    ):
        backend.load()

    mock_transformers.VitPoseForPoseEstimation.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


def test_model_type() -> None:
    from sheaf.api.base import ModelType
    from sheaf.backends.vitpose import ViTPoseBackend

    assert ViTPoseBackend().model_type == ModelType.POSE


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="PoseRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_pose_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, PoseResponse)
    assert len(resp.poses) == 1
    assert len(resp.poses[0]) == 17


def test_predict_keypoints_are_xyz_triples(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    for kpt in resp.poses[0]:
        assert len(kpt) == 3  # [x, y, score]

    x0, y0, s0 = resp.poses[0][0]
    assert x0 == pytest.approx(_KPT_XY_P0[0][0])
    assert y0 == pytest.approx(_KPT_XY_P0[0][1])
    assert s0 == pytest.approx(_KPT_SCORES_P0[0])


def test_predict_multiple_persons(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    persons = [
        _make_person_result(_KPT_XY_P0, _KPT_SCORES_P0),
        _make_person_result(_KPT_XY_P1, _KPT_SCORES_P1),
    ]
    mock = _make_transformers_mod(persons=persons)
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert len(resp.poses) == 2


def test_predict_bboxes_forwarded_to_processor(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    bboxes = [[10.0, 20.0, 200.0, 400.0]]
    mock = _make_transformers_mod()
    _, processor = _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64, bboxes=bboxes)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(req)

    _, kwargs = processor.call_args
    assert kwargs["boxes"] == [bboxes]


def test_predict_defaults_to_full_image_bbox(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    w, h = _IMG_SIZE
    mock = _make_transformers_mod()
    _, processor = _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(req)

    _, kwargs = processor.call_args
    assert kwargs["boxes"] == [[[0, 0, w, h]]]


def test_predict_keypoint_names_from_id2label(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.keypoint_names == _COCO_NAMES


def test_predict_missing_id2label_returns_empty_names(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, processor = _wire(loaded_backend, mock)
    del model.config.id2label
    loaded_backend._Image = _make_pil_mock()[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.keypoint_names == []


def test_predict_image_dimensions(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    img_w, img_h = 1280, 720
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock(size=(img_w, img_h))[1]

    req = PoseRequest(model_name="vitpose", image_b64=_IMG_B64)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(req)

    assert resp.width == img_w
    assert resp.height == img_h


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)
    loaded_backend._Image = _make_pil_mock()[1]

    reqs = [
        PoseRequest(model_name="vitpose", image_b64=_IMG_B64),
        PoseRequest(model_name="vitpose", image_b64=_IMG_B64),
    ]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, PoseResponse) for r in responses)
    assert model.call_count == 2
