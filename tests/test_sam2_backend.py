"""Tests for SAM2Backend — fully mocked, no sam2 or torch required.

Covers:
  - load() raises ImportError when sam2 is absent
  - load() passes correct model_name and device to from_pretrained
  - predict() rejects non-SegmentationRequest inputs
  - predict() with point prompts — correct numpy arrays passed to predictor
  - predict() with box prompt — box passed, no point arrays
  - predict() with both point and box prompts
  - predict() multimask_output=True — three masks returned
  - predict() multimask_output=False — one mask returned
  - predict() mask encoding — masks_b64 round-trips to correct shape + values
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.segmentation import SegmentationRequest, SegmentationResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_H, _W = 64, 64
_IMG_B64 = base64.b64encode(b"fake_image_bytes").decode()


# ---------------------------------------------------------------------------
# Fake sam2 module + predictor factory
# ---------------------------------------------------------------------------


def _make_predictor_mock(
    n_masks: int = 3,
    h: int = _H,
    w: int = _W,
) -> MagicMock:
    masks = np.ones((n_masks, h, w), dtype=bool)
    scores = np.linspace(0.9, 0.7, n_masks).astype(np.float32)
    logits = np.zeros((n_masks, 256, 256), dtype=np.float32)

    predictor = MagicMock()
    predictor.predict.return_value = (masks, scores, logits)
    return predictor


def _make_sam2_mod(predictor: MagicMock | None = None) -> ModuleType:
    if predictor is None:
        predictor = _make_predictor_mock()

    sam2_mod = ModuleType("sam2")
    image_pred_mod = ModuleType("sam2.sam2_image_predictor")

    cls_mock = MagicMock()
    cls_mock.from_pretrained.return_value = predictor
    image_pred_mod.SAM2ImagePredictor = cls_mock  # type: ignore[attr-defined]

    sam2_mod.sam2_image_predictor = image_pred_mod  # type: ignore[attr-defined]
    return sam2_mod


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
def predictor() -> MagicMock:
    return _make_predictor_mock()


@pytest.fixture
def loaded_backend(predictor: MagicMock):  # type: ignore[no-untyped-def]
    from sheaf.backends.sam2 import SAM2Backend

    backend = SAM2Backend(model_name="facebook/sam2.1-hiera-base-plus", device="cpu")
    sam2_mod = _make_sam2_mod(predictor)
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {
            "sam2": sam2_mod,
            "sam2.sam2_image_predictor": sam2_mod.sam2_image_predictor,
            "PIL": pil_mod,
        },
    ):
        backend.load()
    # Wire the predictor directly so tests can assert on it
    backend._predictor = predictor
    backend._Image = pil_mod.Image
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_sam2() -> None:
    import builtins

    from sheaf.backends.sam2 import SAM2Backend

    backend = SAM2Backend()
    pil_mod, _ = _make_pil_mock()
    mods_without = {k: v for k, v in sys.modules.items() if not k.startswith("sam2")}
    mods_without["PIL"] = pil_mod

    _real_import = builtins.__import__

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "sam2" or name.startswith("sam2."):
            raise ModuleNotFoundError("No module named 'sam2'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[vision\\]"),
    ):
        backend.load()


def test_load_passes_correct_model_name() -> None:
    from sheaf.backends.sam2 import SAM2Backend

    backend = SAM2Backend(model_name="facebook/sam2.1-hiera-large")
    sam2_mod = _make_sam2_mod()
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {
            "sam2": sam2_mod,
            "sam2.sam2_image_predictor": sam2_mod.sam2_image_predictor,
            "PIL": pil_mod,
        },
    ):
        backend.load()

    sam2_mod.sam2_image_predictor.SAM2ImagePredictor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "facebook/sam2.1-hiera-large", device="cpu"
    )


def test_load_passes_device() -> None:
    from sheaf.backends.sam2 import SAM2Backend

    backend = SAM2Backend(device="cuda")
    sam2_mod = _make_sam2_mod()
    pil_mod, _ = _make_pil_mock()
    with patch.dict(
        sys.modules,
        {
            "sam2": sam2_mod,
            "sam2.sam2_image_predictor": sam2_mod.sam2_image_predictor,
            "PIL": pil_mod,
        },
    ):
        backend.load()

    sam2_mod.sam2_image_predictor.SAM2ImagePredictor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "facebook/sam2.1-hiera-base-plus", device="cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_non_segmentation_request(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="SegmentationRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — prompts passed correctly
# ---------------------------------------------------------------------------


def test_predict_with_point_prompts(loaded_backend, predictor: MagicMock) -> None:  # type: ignore[no-untyped-def]
    req = SegmentationRequest(
        model_name="sam2",
        image_b64=_IMG_B64,
        point_coords=[[100.0, 200.0], [150.0, 250.0]],
        point_labels=[1, 0],
    )
    resp = loaded_backend.predict(req)

    assert isinstance(resp, SegmentationResponse)
    _, kwargs = predictor.predict.call_args
    expected_coords = [[100.0, 200.0], [150.0, 250.0]]
    np.testing.assert_array_equal(kwargs["point_coords"], expected_coords)
    np.testing.assert_array_equal(kwargs["point_labels"], [1, 0])
    assert kwargs["box"] is None


def test_predict_with_box_prompt(loaded_backend, predictor: MagicMock) -> None:  # type: ignore[no-untyped-def]
    req = SegmentationRequest(
        model_name="sam2",
        image_b64=_IMG_B64,
        box=[10.0, 20.0, 300.0, 400.0],
    )
    resp = loaded_backend.predict(req)

    assert isinstance(resp, SegmentationResponse)
    _, kwargs = predictor.predict.call_args
    np.testing.assert_array_equal(kwargs["box"], [10.0, 20.0, 300.0, 400.0])
    assert kwargs["point_coords"] is None
    assert kwargs["point_labels"] is None


def test_predict_with_point_and_box(loaded_backend, predictor: MagicMock) -> None:  # type: ignore[no-untyped-def]
    req = SegmentationRequest(
        model_name="sam2",
        image_b64=_IMG_B64,
        point_coords=[[50.0, 50.0]],
        point_labels=[1],
        box=[0.0, 0.0, 200.0, 200.0],
    )
    resp = loaded_backend.predict(req)

    assert isinstance(resp, SegmentationResponse)
    _, kwargs = predictor.predict.call_args
    assert kwargs["point_coords"] is not None
    assert kwargs["box"] is not None


# ---------------------------------------------------------------------------
# predict() — multimask_output
# ---------------------------------------------------------------------------


def test_predict_multimask_true(loaded_backend, predictor: MagicMock) -> None:  # type: ignore[no-untyped-def]
    req = SegmentationRequest(
        model_name="sam2",
        image_b64=_IMG_B64,
        box=[0.0, 0.0, 100.0, 100.0],
        multimask_output=True,
    )
    resp = loaded_backend.predict(req)

    assert len(resp.masks_b64) == 3
    assert len(resp.scores) == 3
    _, kwargs = predictor.predict.call_args
    assert kwargs["multimask_output"] is True


def test_predict_multimask_false(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    # Return a single mask from the predictor
    one_mask = np.ones((1, _H, _W), dtype=bool)
    one_score = np.array([0.95], dtype=np.float32)
    loaded_backend._predictor.predict.return_value = (
        one_mask,
        one_score,
        np.zeros((1, 256, 256), dtype=np.float32),
    )

    req = SegmentationRequest(
        model_name="sam2",
        image_b64=_IMG_B64,
        box=[0.0, 0.0, 100.0, 100.0],
        multimask_output=False,
    )
    resp = loaded_backend.predict(req)

    assert len(resp.masks_b64) == 1
    assert len(resp.scores) == 1
    _, kwargs = loaded_backend._predictor.predict.call_args
    assert kwargs["multimask_output"] is False


# ---------------------------------------------------------------------------
# predict() — mask encoding round-trip
# ---------------------------------------------------------------------------


def test_predict_mask_encoding_round_trip(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """masks_b64 decodes back to the original boolean mask."""
    known = np.zeros((_H, _W), dtype=bool)
    known[10:20, 10:20] = True  # small square region
    loaded_backend._predictor.predict.return_value = (
        known[np.newaxis],  # shape (1, H, W)
        np.array([0.88], dtype=np.float32),
        np.zeros((1, 256, 256), dtype=np.float32),
    )

    req = SegmentationRequest(
        model_name="sam2",
        image_b64=_IMG_B64,
        box=[0.0, 0.0, 64.0, 64.0],
        multimask_output=False,
    )
    resp = loaded_backend.predict(req)

    assert resp.height == _H
    assert resp.width == _W

    decoded = (
        np.frombuffer(base64.b64decode(resp.masks_b64[0]), dtype=np.uint8)
        .reshape(resp.height, resp.width)
        .astype(bool)
    )

    np.testing.assert_array_equal(decoded, known)


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend, predictor: MagicMock) -> None:  # type: ignore[no-untyped-def]
    reqs = [
        SegmentationRequest(
            model_name="sam2",
            image_b64=_IMG_B64,
            box=[0.0, 0.0, 32.0, 32.0],
        ),
        SegmentationRequest(
            model_name="sam2",
            image_b64=_IMG_B64,
            point_coords=[[16.0, 16.0]],
            point_labels=[1],
        ),
    ]
    responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, SegmentationResponse) for r in responses)
    # set_image called once per request
    assert predictor.set_image.call_count == 2
