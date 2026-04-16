"""API contract for cross-modal embedding models (ImageBind, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import field_validator, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType

# Canonical modality names matching ImageBind's ModalityType values.
MODALITY_TEXT = "text"
MODALITY_VISION = "vision"
MODALITY_AUDIO = "audio"
MODALITY_DEPTH = "depth"
MODALITY_THERMAL = "thermal"


class MultimodalEmbeddingRequest(BaseRequest):
    """Request contract for cross-modal embedding models (e.g. ImageBind).

    Exactly one modality field must be set per request.  All items in the
    chosen field are embedded in a single forward pass and returned in the
    shared embedding space.

    Modalities:
        texts:             List of strings (text modality).
        images_b64:        List of base64-encoded image files (vision modality).
        audios_b64:        List of base64-encoded audio files (audio modality).
        depth_images_b64:  List of base64-encoded depth images (depth modality).
        thermal_images_b64: List of base64-encoded thermal images (thermal modality).

    For image/audio inputs any format the underlying model accepts is valid
    (JPEG/PNG for vision; WAV/MP3 for audio).  The backend writes temporary
    files as needed — the model loaders read paths, not raw bytes.

    Args:
        normalize: If True (default), L2-normalize output embeddings so that
            cosine similarity equals dot product.
    """

    model_type: Literal[ModelType.MULTIMODAL_EMBEDDING] = ModelType.MULTIMODAL_EMBEDDING

    texts: list[str] | None = None
    images_b64: list[str] | None = None
    audios_b64: list[str] | None = None
    depth_images_b64: list[str] | None = None
    thermal_images_b64: list[str] | None = None
    normalize: bool = True

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator(
        "images_b64", "audios_b64", "depth_images_b64", "thermal_images_b64"
    )
    @classmethod
    def _validate_base64(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for item in v:
                try:
                    base64.b64decode(item, validate=True)
                except Exception as e:
                    raise ValueError(
                        "All base64 fields must contain valid base64-encoded bytes"
                    ) from e
        return v

    @model_validator(mode="after")
    def validate_exactly_one_modality(self) -> MultimodalEmbeddingRequest:
        active = [
            name
            for name, val in [
                ("texts", self.texts),
                ("images_b64", self.images_b64),
                ("audios_b64", self.audios_b64),
                ("depth_images_b64", self.depth_images_b64),
                ("thermal_images_b64", self.thermal_images_b64),
            ]
            if val is not None
        ]
        if len(active) == 0:
            raise ValueError(
                "Exactly one modality field must be provided. "
                "Options: texts, images_b64, audios_b64, "
                "depth_images_b64, thermal_images_b64."
            )
        if len(active) > 1:
            raise ValueError(
                f"Only one modality field may be set per request, got: {active}"
            )
        return self

    @property
    def modality(self) -> str:
        """Return the canonical modality name for the active field."""
        if self.texts is not None:
            return MODALITY_TEXT
        if self.images_b64 is not None:
            return MODALITY_VISION
        if self.audios_b64 is not None:
            return MODALITY_AUDIO
        if self.depth_images_b64 is not None:
            return MODALITY_DEPTH
        if self.thermal_images_b64 is not None:
            return MODALITY_THERMAL
        raise RuntimeError("No modality set — should have been caught by validator")

    @property
    def n_items(self) -> int:
        """Number of items in the active modality field."""
        for val in [
            self.texts,
            self.images_b64,
            self.audios_b64,
            self.depth_images_b64,
            self.thermal_images_b64,
        ]:
            if val is not None:
                return len(val)
        raise RuntimeError("No modality set — should have been caught by validator")


class MultimodalEmbeddingResponse(BaseResponse):
    """Response contract for cross-modal embedding models."""

    model_type: Literal[ModelType.MULTIMODAL_EMBEDDING] = ModelType.MULTIMODAL_EMBEDDING

    # Embedding matrix — shape (N, dim) where N = number of inputs
    embeddings: list[list[float]]

    # Embedding dimensionality (1024 for ImageBind)
    dim: int

    # Which modality produced these embeddings
    modality: str
