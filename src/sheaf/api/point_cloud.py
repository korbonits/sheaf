"""API contract for 3D point cloud models (PointNet, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import Field, field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class PointCloudRequest(BaseRequest):
    """Request contract for 3D point cloud processing.

    Point clouds are passed as base64-encoded flat float32 byte arrays of shape
    ``(n_points, 3)`` containing XYZ coordinates.  Points are expected to be
    pre-normalised to a unit sphere centred at the origin (subtract centroid,
    divide by max radius).

    Args:
        points_b64: Base64-encoded flat float32 byte array, shape ``(n_points, 3)``.
            Decode with::

                pts = np.frombuffer(
                    base64.b64decode(points_b64), dtype=np.float32
                ).reshape(n_points, 3)

        n_points: Number of points in the cloud.  Required to reshape the flat
            byte array.  Typical values: 1024, 2048, 4096.
        task: "embed" — return the 1024-dim PointNet global feature vector.
              "classify" — return class label + per-class softmax scores.
    """

    model_type: Literal[ModelType.POINT_CLOUD] = ModelType.POINT_CLOUD

    points_b64: str
    n_points: int = Field(ge=1)
    task: Literal["embed", "classify"] = "embed"

    @field_validator("points_b64")
    @classmethod
    def validate_points_b64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("points_b64 must be valid base64-encoded bytes") from e
        return v


class PointCloudResponse(BaseResponse):
    """Response contract for 3D point cloud processing.

    Exactly one of ``embedding`` or ``labels`` is populated, depending on the
    requested ``task``.

    For ``task="embed"``:
        ``embedding`` — 1024-dim global PointNet feature (L2-normalised).

    For ``task="classify"``:
        ``label``       — top predicted class name (e.g. "airplane").
        ``scores``      — per-class softmax probabilities, parallel to ``label_names``.
        ``label_names`` — class names in score order (model's id2label mapping).
    """

    model_type: Literal[ModelType.POINT_CLOUD] = ModelType.POINT_CLOUD

    # task="embed"
    embedding: list[float] | None = None

    # task="classify"
    label: str | None = None
    scores: list[float] | None = None
    label_names: list[str] | None = None
