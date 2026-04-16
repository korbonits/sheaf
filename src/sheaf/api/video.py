"""API contract for video understanding models (VideoMAE, TimeSformer, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class VideoRequest(BaseRequest):
    """Request contract for video understanding models.

    Frames are passed as a list of base64-encoded images (JPEG or PNG).
    The number of frames expected depends on the model:

    - VideoMAE-base:  16 frames (default, tubelet_size=2, 224×224)
    - TimeSformer:    8 frames  (224×224)

    Pass exactly the number the model was pretrained on, or the processor
    will pad/truncate automatically.

    Args:
        frames_b64: Ordered list of base64-encoded video frames.
        task: "embedding" returns a single fixed-size vector per video clip;
            "classification" returns class labels and softmax scores.
        pooling: Pooling strategy for embeddings.
            "cls"  — CLS token at position 0 of last_hidden_state (default).
            "mean" — Mean of all non-CLS patch tokens.
        normalize: If True (default), L2-normalize the output embedding.
            Ignored for classification.
    """

    model_type: Literal[ModelType.VIDEO] = ModelType.VIDEO

    frames_b64: list[str] = Field(min_length=1)
    task: Literal["embedding", "classification"] = "embedding"
    pooling: Literal["cls", "mean"] = "cls"
    normalize: bool = True

    @field_validator("frames_b64")
    @classmethod
    def validate_frames_base64(cls, v: list[str]) -> list[str]:
        for item in v:
            try:
                base64.b64decode(item, validate=True)
            except Exception as e:
                raise ValueError(
                    "frames_b64 items must be valid base64-encoded bytes"
                ) from e
        return v


class VideoResponse(BaseResponse):
    """Response contract for video understanding models.

    For ``task="embedding"``: ``embedding`` and ``dim`` are populated.
    For ``task="classification"``: ``labels`` and ``scores`` are populated.
    """

    model_type: Literal[ModelType.VIDEO] = ModelType.VIDEO

    task: str

    # Embedding task
    embedding: list[float] | None = None
    dim: int | None = None

    # Classification task
    labels: list[str] | None = None
    scores: list[float] | None = None

    @model_validator(mode="after")
    def validate_task_fields(self) -> VideoResponse:
        if self.task == "embedding":
            if self.embedding is None or self.dim is None:
                raise ValueError(
                    "embedding task response must include 'embedding' and 'dim'"
                )
        elif self.task == "classification":
            if self.labels is None or self.scores is None:
                raise ValueError(
                    "classification task response must include 'labels' and 'scores'"
                )
        return self
