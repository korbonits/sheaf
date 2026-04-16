"""API contract for diffusion image generation models (FLUX, etc.)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class DiffusionRequest(BaseRequest):
    """Request contract for text-to-image diffusion models.

    Args:
        prompt: Text description of the image to generate.
        negative_prompt: Text description of what to avoid.  Not supported
            by all models (FLUX.1-schnell ignores it).
        height: Output image height in pixels.  Must be a multiple of 8.
            Defaults to 1024.
        width: Output image width in pixels.  Must be a multiple of 8.
            Defaults to 1024.
        num_inference_steps: Number of denoising steps.  FLUX.1-schnell
            is optimized for 1–4 steps; FLUX.1-dev typically uses 20–50.
        guidance_scale: Classifier-free guidance scale.  Higher values
            steer generation closer to the prompt.  FLUX.1-schnell uses
            0.0 (guidance-distilled); FLUX.1-dev typically uses 3.5–7.0.
        seed: Random seed for reproducibility.  None = random.
    """

    model_type: Literal[ModelType.DIFFUSION] = ModelType.DIFFUSION

    prompt: str
    negative_prompt: str = ""
    height: int = Field(default=1024, ge=64, multiple_of=8)
    width: int = Field(default=1024, ge=64, multiple_of=8)
    num_inference_steps: int = Field(default=4, ge=1, le=200)
    guidance_scale: float = Field(default=0.0, ge=0.0)
    seed: int | None = None

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v


class DiffusionResponse(BaseResponse):
    """Response contract for text-to-image diffusion models.

    The generated image is returned as a base64-encoded PNG.  To decode::

        import base64
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(base64.b64decode(image_b64)))

    Args:
        image_b64: Base64-encoded PNG image.
        height: Output image height in pixels.
        width: Output image width in pixels.
        seed: Seed actually used for generation (useful when the request
            seed was None and you want to reproduce the result).
    """

    model_type: Literal[ModelType.DIFFUSION] = ModelType.DIFFUSION

    image_b64: str
    height: int
    width: int
    seed: int
