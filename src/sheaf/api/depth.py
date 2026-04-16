"""API contract for monocular depth estimation models (Depth Anything v2, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class DepthRequest(BaseRequest):
    """Request contract for monocular depth estimation.

    Args:
        image_b64: Base64-encoded image file.  Any format PIL can open is
            accepted (JPEG, PNG, WebP, etc.).
        normalize: If True (default), the depth map is linearly rescaled to
            [0, 1] where 0 = nearest and 1 = furthest point in the scene.
            If False, raw relative depth values from the model are returned.
    """

    model_type: Literal[ModelType.DEPTH] = ModelType.DEPTH

    image_b64: str
    normalize: bool = True

    @field_validator("image_b64")
    @classmethod
    def validate_image_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("image_b64 must be a valid base64-encoded string") from e
        return v


class DepthResponse(BaseResponse):
    """Response contract for monocular depth estimation.

    The depth map is a base64-encoded flat float32 byte array at the model's
    native output resolution.  To reconstruct::

        import base64, numpy as np
        depth = np.frombuffer(
            base64.b64decode(depth_b64), dtype=np.float32
        ).reshape(height, width)

    If ``normalize=True`` was requested, values are in [0, 1].
    ``min_depth`` and ``max_depth`` are the raw (pre-normalization) bounds,
    useful for recovering metric-relative scale.
    """

    model_type: Literal[ModelType.DEPTH] = ModelType.DEPTH

    depth_b64: str
    height: int
    width: int

    # Raw depth bounds before any normalization
    min_depth: float
    max_depth: float
