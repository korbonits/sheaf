"""API contract for pose estimation models (ViTPose, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class PoseRequest(BaseRequest):
    """Request contract for human pose estimation.

    ViTPose is a top-down model: it estimates keypoints within person crops.
    If ``bboxes`` is provided, each box is used as a person crop. If omitted,
    the full image is treated as a single-person crop.

    Args:
        image_b64: Base64-encoded image (JPEG, PNG, or any PIL-readable format).
        bboxes: Optional list of person bounding boxes in pixel coordinates,
            each ``[x_min, y_min, x_max, y_max]``. If None, defaults to the
            full image as one person crop.
        threshold: Minimum keypoint confidence score to include in the response.
            Keypoints below this threshold are still returned but flagged by a
            low score; filtering is left to the caller.
    """

    model_type: Literal[ModelType.POSE] = ModelType.POSE

    image_b64: str
    bboxes: list[list[float]] | None = None
    threshold: float = 0.3

    @field_validator("image_b64")
    @classmethod
    def validate_image_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("image_b64 must be a valid base64-encoded string") from e
        return v


class PoseResponse(BaseResponse):
    """Response contract for human pose estimation.

    ``poses[i][j]`` is ``[x, y, score]`` for the j-th keypoint of the i-th
    detected person, in absolute pixel coordinates.  ``keypoint_names[j]``
    gives the semantic label for keypoint j (e.g. "nose", "left_eye").

    Decode example::

        for person in resp.poses:
            for (x, y, score), name in zip(person, resp.keypoint_names):
                print(f"{name}: ({x:.1f}, {y:.1f})  conf={score:.2f}")
    """

    model_type: Literal[ModelType.POSE] = ModelType.POSE

    # poses[person][keypoint] = [x, y, score]
    poses: list[list[list[float]]]

    # Semantic keypoint names parallel to the keypoint axis.
    keypoint_names: list[str]

    # Original image dimensions in pixels.
    width: int
    height: int
