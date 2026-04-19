"""API contract for optical flow models (RAFT, UniMatch, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class OpticalFlowRequest(BaseRequest):
    """Request contract for optical flow estimation.

    Accepts two consecutive video frames and returns the dense per-pixel
    displacement field between them.

    Args:
        frame1_b64: Base64-encoded first frame (JPEG, PNG, or any PIL-readable
            format). Both frames must have the same spatial dimensions.
        frame2_b64: Base64-encoded second frame.
    """

    model_type: Literal[ModelType.OPTICAL_FLOW] = ModelType.OPTICAL_FLOW

    frame1_b64: str
    frame2_b64: str

    @field_validator("frame1_b64", "frame2_b64")
    @classmethod
    def validate_frame_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("frame must be valid base64-encoded bytes") from e
        return v


class OpticalFlowResponse(BaseResponse):
    """Response contract for optical flow estimation.

    ``flow_b64`` is a base64-encoded flat float32 byte array of shape
    ``(height, width, 2)``, where the last dimension is ``(dx, dy)`` —
    the horizontal and vertical pixel displacement from frame1 to frame2.

    Decode example::

        import base64, numpy as np
        flow = np.frombuffer(
            base64.b64decode(flow_b64), dtype=np.float32
        ).reshape(height, width, 2)
        dx, dy = flow[..., 0], flow[..., 1]
    """

    model_type: Literal[ModelType.OPTICAL_FLOW] = ModelType.OPTICAL_FLOW

    # Base64-encoded float32 array, shape (height, width, 2).
    flow_b64: str

    # Spatial dimensions of the output flow field (matches input frame size).
    width: int
    height: int
