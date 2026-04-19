"""SDXL backend for text+image-conditioned generation via HuggingFace diffusers.

Requires: pip install "sheaf-serve[multimodal-generation]"
Models: "stabilityai/stable-diffusion-xl-base-1.0" (default)

Two operating modes selected at backend-construction time:

  mode="img2img"  — StableDiffusionXLImg2ImgPipeline
      Takes a prompt + conditioning image.  ``strength`` controls how much
      the output departs from the input (0 = identical, 1 = ignore original).

  mode="inpaint"  — StableDiffusionXLInpaintPipeline
      Takes a prompt + image + mask.  White mask pixels are regenerated;
      black pixels are preserved.

PIL.Image is stored as ``self._Image`` at load() time for test injectability,
following the same pattern as ViTPoseBackend and DepthAnythingBackend.

torch_dtype is stored as a string at __init__ time and resolved to the actual
torch.dtype inside load(), following the same pattern as FluxBackend.
"""

from __future__ import annotations

import base64
import io
import random
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.multimodal_generation import (
    MultimodalGenerationRequest,
    MultimodalGenerationResponse,
)
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_VALID_MODES = {"img2img", "inpaint"}


@register_backend("sdxl")
class SDXLBackend(ModelBackend):
    """ModelBackend for Stable Diffusion XL img2img and inpainting.

    Args:
        model_id: HuggingFace model ID.
            "stabilityai/stable-diffusion-xl-base-1.0"  (default)
        mode: Pipeline mode.
            "img2img"  — text + image conditioning (default)
            "inpaint"  — text + image + mask inpainting
        device: "cpu", "cuda", or "mps".
        torch_dtype: "bfloat16" (default), "float16", or "float32".
        enable_model_cpu_offload: Offload model components to CPU between
            uses to reduce peak VRAM.  When True, ``.to(device)`` is skipped.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        mode: str = "img2img",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        enable_model_cpu_offload: bool = False,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self._model_id = model_id
        self._mode = mode
        self._device = device
        self._torch_dtype_str = torch_dtype
        self._enable_cpu_offload = enable_model_cpu_offload
        self._pipeline: Any = None
        self._Image: Any = None  # PIL.Image — injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.MULTIMODAL_GENERATION

    def load(self) -> None:
        try:
            import torch  # ty: ignore[unresolved-import]
            from diffusers import (  # ty: ignore[unresolved-import]
                StableDiffusionXLImg2ImgPipeline,
                StableDiffusionXLInpaintPipeline,
            )
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "diffusers, torch, and Pillow are required for the SDXL backend. "
                "Install with: pip install 'sheaf-serve[multimodal-generation]'"
            ) from e

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self._torch_dtype_str, torch.bfloat16)

        pipeline_cls = (
            StableDiffusionXLImg2ImgPipeline
            if self._mode == "img2img"
            else StableDiffusionXLInpaintPipeline
        )
        self._pipeline = pipeline_cls.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
        )

        if self._enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to(self._device)

        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, MultimodalGenerationRequest):
            raise TypeError(
                f"Expected MultimodalGenerationRequest, got {type(request)}"
            )
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(
        self, request: MultimodalGenerationRequest
    ) -> MultimodalGenerationResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        generator = torch.Generator(device=self._device).manual_seed(seed)

        cond_image = self._Image.open(
            io.BytesIO(base64.b64decode(request.image_b64))
        ).convert("RGB")

        call_kwargs: dict[str, Any] = dict(
            prompt=request.prompt,
            image=cond_image,
            strength=request.strength,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        if request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt
        if request.mask_b64 is not None:
            call_kwargs["mask_image"] = self._Image.open(
                io.BytesIO(base64.b64decode(request.mask_b64))
            ).convert("RGB")

        result = self._pipeline(**call_kwargs)
        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        return MultimodalGenerationResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            image_b64=image_b64,
            width=image.width,
            height=image.height,
            seed=seed,
        )
