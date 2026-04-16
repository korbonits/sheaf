"""API contract for image segmentation models (SAM2, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class SegmentationRequest(BaseRequest):
    """Request contract for prompted image segmentation.

    Exactly one image is segmented per request.  At least one prompt must be
    provided — either ``point_coords`` (with matching ``point_labels``) or
    ``box``, or both.

    Args:
        image_b64: Base64-encoded image file.  Any format PIL can open is
            accepted (JPEG, PNG, WebP, etc.).
        point_coords: List of [x, y] points (pixel coordinates).
        point_labels: Foreground (1) / background (0) label for each point.
            Must have the same length as ``point_coords``.
        box: Bounding-box prompt as [x1, y1, x2, y2] in pixel coordinates.
        multimask_output: If True (default), return three candidate masks
            ranked by score.  Set to False to get a single best mask.
    """

    model_type: Literal[ModelType.SEGMENTATION] = ModelType.SEGMENTATION

    image_b64: str
    point_coords: list[list[float]] | None = None
    point_labels: list[int] | None = None
    box: list[float] | None = None
    multimask_output: bool = True

    @field_validator("image_b64")
    @classmethod
    def validate_image_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("image_b64 must be a valid base64-encoded string") from e
        return v

    @model_validator(mode="after")
    def validate_prompts(self) -> SegmentationRequest:
        if self.point_coords is None and self.box is None:
            raise ValueError(
                "At least one prompt is required: provide 'point_coords' or 'box'"
            )
        if self.point_coords is not None:
            if self.point_labels is None:
                raise ValueError(
                    "'point_labels' is required when 'point_coords' is provided"
                )
            if len(self.point_coords) != len(self.point_labels):
                raise ValueError(
                    f"'point_coords' has {len(self.point_coords)} entries but "
                    f"'point_labels' has {len(self.point_labels)}"
                )
        return self


class SegmentationResponse(BaseResponse):
    """Response contract for image segmentation models.

    Each mask is a base64-encoded flat uint8 byte array.  To reconstruct::

        import base64, numpy as np
        mask = np.frombuffer(
            base64.b64decode(masks_b64[i]), dtype=np.uint8
        ).reshape(height, width).astype(bool)
    """

    model_type: Literal[ModelType.SEGMENTATION] = ModelType.SEGMENTATION

    # One entry per output mask, in descending score order.
    masks_b64: list[str]
    scores: list[float]

    # Original image dimensions — needed to reshape masks_b64.
    height: int
    width: int
