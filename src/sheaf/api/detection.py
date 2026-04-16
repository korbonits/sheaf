"""API contract for object detection models (DETR, RT-DETR, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class DetectionRequest(BaseRequest):
    """Request contract for object detection.

    Args:
        image_b64: Base64-encoded image file.  Any format PIL can open is
            accepted (JPEG, PNG, WebP, etc.).
        threshold: Minimum confidence score for a detection to be included
            in the response.  Defaults to 0.5.
    """

    model_type: Literal[ModelType.DETECTION] = ModelType.DETECTION

    image_b64: str
    threshold: float = 0.5

    @field_validator("image_b64")
    @classmethod
    def validate_image_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("image_b64 must be a valid base64-encoded string") from e
        return v


class DetectionResponse(BaseResponse):
    """Response contract for object detection.

    Boxes are in absolute pixel coordinates: ``[x_min, y_min, x_max, y_max]``.
    Lists are parallel — ``boxes[i]``, ``scores[i]``, and ``labels[i]`` all
    describe the same detection, sorted by descending confidence score.
    """

    model_type: Literal[ModelType.DETECTION] = ModelType.DETECTION

    boxes: list[list[float]]  # [[x1, y1, x2, y2], ...]
    scores: list[float]
    labels: list[str]

    # Original image dimensions
    width: int
    height: int
