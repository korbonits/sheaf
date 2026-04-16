"""API contract for embedding / representation models (CLIP, DINOv2, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class EmbeddingRequest(BaseRequest):
    """Request contract for embedding models.

    Exactly one of ``texts`` or ``images_b64`` must be provided per request.
    Both fields accept a batch — pass multiple items to embed them in a single
    forward pass.

    Args:
        texts: List of strings to embed (text modality).
        images_b64: List of base64-encoded image files to embed (vision modality).
            Any format PIL can open is accepted (JPEG, PNG, WebP, etc.).
        normalize: If True (default), L2-normalize the output embeddings so that
            cosine similarity equals dot product.
    """

    model_type: Literal[ModelType.EMBEDDING] = ModelType.EMBEDDING

    texts: list[str] | None = None
    images_b64: list[str] | None = None
    normalize: bool = True

    @field_validator("images_b64")
    @classmethod
    def validate_images_base64(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for item in v:
                try:
                    base64.b64decode(item, validate=True)
                except Exception as e:
                    raise ValueError(
                        "images_b64 items must be valid base64-encoded bytes"
                    ) from e
        return v

    @model_validator(mode="after")
    def validate_exactly_one_input(self) -> EmbeddingRequest:
        if self.texts is None and self.images_b64 is None:
            raise ValueError("Exactly one of 'texts' or 'images_b64' must be provided")
        if self.texts is not None and self.images_b64 is not None:
            raise ValueError("Provide either 'texts' or 'images_b64', not both")
        return self


class EmbeddingResponse(BaseResponse):
    """Response contract for embedding models."""

    model_type: Literal[ModelType.EMBEDDING] = ModelType.EMBEDDING

    # Embedding matrix — shape (N, dim) where N = number of inputs
    embeddings: list[list[float]]

    # Embedding dimensionality (e.g. 512 for ViT-B-32)
    dim: int
