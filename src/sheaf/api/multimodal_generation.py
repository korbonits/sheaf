"""API contract for text+image-conditioned generation models (SDXL, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class MultimodalGenerationRequest(BaseRequest):
    """Request contract for text+image-conditioned image generation.

    Distinct from pure text-to-image (DiffusionRequest/FLUX): the input image
    conditions the generation.  When ``mask_b64`` is omitted the backend runs
    img2img (style/content transfer); when provided it runs inpainting.

    Args:
        prompt: Text description guiding the generated image.
        image_b64: Base64-encoded input image (JPEG, PNG, or any PIL-readable
            format).  Acts as the conditioning source for img2img / inpainting.
        mask_b64: Optional base64-encoded mask image (same spatial size as
            ``image_b64``).  White pixels are re-generated; black pixels are
            preserved.  Only used when the backend is in ``inpaint`` mode.
        strength: How much to transform the input image.  0.0 = no change,
            1.0 = ignore the original image entirely.  Default 0.8.
        num_inference_steps: Total denoising steps.  Actual steps run =
            ``round(strength * num_inference_steps)``.  Default 50.
        guidance_scale: Classifier-free guidance scale.  Higher values steer
            generation closer to the prompt.  Default 7.5.
        negative_prompt: Text description of what to avoid in the output.
        seed: Random seed for reproducibility.  None = random.
        adapters: Names of LoRA adapters to apply, in order.  Each name must
            be registered on the deployment's ``ModelSpec.lora.adapters``.
            Empty (default) means the deployment default is used (or no LoRA
            if no default is set).
        adapter_weights: Per-adapter weights, parallel to ``adapters``.  When
            ``None`` (default), the per-adapter ``weight`` from
            ``LoRAConfig.adapters[name]`` is used.  Length must match
            ``adapters`` when provided.
    """

    model_type: Literal[ModelType.MULTIMODAL_GENERATION] = (
        ModelType.MULTIMODAL_GENERATION
    )

    prompt: str
    image_b64: str
    mask_b64: str | None = None
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    num_inference_steps: int = Field(default=50, ge=1, le=500)
    guidance_scale: float = Field(default=7.5, ge=0.0)
    negative_prompt: str = ""
    seed: int | None = None
    adapters: list[str] = Field(default_factory=list)
    adapter_weights: list[float] | None = None

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("image_b64")
    @classmethod
    def validate_image_b64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("image_b64 must be valid base64-encoded bytes") from e
        return v

    @field_validator("mask_b64")
    @classmethod
    def validate_mask_b64(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("mask_b64 must be valid base64-encoded bytes") from e
        return v

    @model_validator(mode="after")
    def _validate_adapters(self) -> MultimodalGenerationRequest:
        if self.adapter_weights is not None:
            if not self.adapters:
                raise ValueError("adapter_weights provided but adapters is empty")
            if len(self.adapter_weights) != len(self.adapters):
                raise ValueError(
                    f"adapter_weights length ({len(self.adapter_weights)}) "
                    f"must match adapters length ({len(self.adapters)})"
                )
        return self


class MultimodalGenerationResponse(BaseResponse):
    """Response contract for text+image-conditioned image generation.

    The generated image is returned as a base64-encoded PNG.  To decode::

        import base64, io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    """

    model_type: Literal[ModelType.MULTIMODAL_GENERATION] = (
        ModelType.MULTIMODAL_GENERATION
    )

    image_b64: str
    width: int
    height: int
    seed: int
