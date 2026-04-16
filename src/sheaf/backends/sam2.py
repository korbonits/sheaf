"""SAM2 backend for prompted image segmentation via Meta's sam2 library.

Requires: pip install "sheaf-serve[vision]"
Library: sam2 (https://github.com/facebookresearch/segment-anything-2)

Supported models (HuggingFace Hub):
  "facebook/sam2.1-hiera-tiny"       — fastest, lowest memory
  "facebook/sam2.1-hiera-small"      — fast, small
  "facebook/sam2.1-hiera-base-plus"  — balanced (default)
  "facebook/sam2.1-hiera-large"      — highest quality

Prompt types:
  point_coords + point_labels — click points (foreground=1, background=0)
  box                         — bounding box [x1, y1, x2, y2]
  both                        — combined point + box prompt

With multimask_output=True (default), three candidate masks are returned
ranked by confidence score.  Set multimask_output=False for a single mask.

PIL.Image is stored as an instance attribute at load() time so the
dependency stays lazy and tests can inject a mock without PIL installed.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.segmentation import SegmentationRequest, SegmentationResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("sam2")
class SAM2Backend(ModelBackend):
    """ModelBackend for SAM2 prompted image segmentation.

    Each request segments a single image.  Pass point and/or box prompts;
    the backend returns binary masks encoded as base64 uint8 byte arrays.

    Args:
        model_name: HuggingFace model ID. Options:
            "facebook/sam2.1-hiera-tiny"
            "facebook/sam2.1-hiera-small"
            "facebook/sam2.1-hiera-base-plus"  (default)
            "facebook/sam2.1-hiera-large"
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "facebook/sam2.1-hiera-base-plus",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._predictor: Any = None
        self._Image: Any = None  # PIL.Image, injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.SEGMENTATION

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from sam2.sam2_image_predictor import (  # ty: ignore[unresolved-import]
                SAM2ImagePredictor,
            )
        except ImportError as e:
            raise ImportError(
                "sam2 is required for the SAM2 backend. "
                "Install it with: pip install 'sheaf-serve[vision]'"
            ) from e
        self._predictor = SAM2ImagePredictor.from_pretrained(
            self._model_name, device=self._device
        )
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, SegmentationRequest):
            raise TypeError(f"Expected SegmentationRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: SegmentationRequest) -> SegmentationResponse:
        if self._predictor is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        img = self._Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert(
            "RGB"
        )

        point_coords: np.ndarray | None = None
        point_labels: np.ndarray | None = None
        box: np.ndarray | None = None

        if request.point_coords is not None:
            point_coords = np.array(request.point_coords, dtype=np.float32)
            point_labels = np.array(request.point_labels, dtype=np.int32)
        if request.box is not None:
            box = np.array(request.box, dtype=np.float32)

        self._predictor.set_image(img)
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=request.multimask_output,
        )
        # masks:  (num_masks, H, W)  bool
        # scores: (num_masks,)       float32

        height: int = masks.shape[1]
        width: int = masks.shape[2]
        masks_b64 = [
            base64.b64encode(mask.astype(np.uint8).tobytes()).decode() for mask in masks
        ]

        return SegmentationResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            masks_b64=masks_b64,
            scores=scores.tolist(),
            height=height,
            width=width,
        )
