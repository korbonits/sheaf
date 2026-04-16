"""FLUX diffusion backend via HuggingFace diffusers.

Supports FLUX.1-schnell (fast, Apache 2.0) and FLUX.1-dev (higher quality,
non-commercial).  Both are single-stage flow-matching models — no VAE
encode/decode round-trip, no CLIP text encoder.

Key characteristics:
- FLUX.1-schnell: 1–4 steps, guidance_scale=0.0 (guidance-distilled)
- FLUX.1-dev: 20–50 steps, guidance_scale=3.5–7.0
- Output: 1024×1024 PNG by default; any multiple-of-8 resolution works
- Memory: ~24 GB fp32, ~12 GB bf16, ~8 GB fp8 (with quantization)

Install:
    pip install 'sheaf-serve[diffusion]'

Usage::

    spec = ModelSpec(
        name="flux-schnell",
        model_type=ModelType.DIFFUSION,
        backend="flux",
        backend_kwargs={
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "device": "cuda",
            "torch_dtype": "bfloat16",
        },
        resources=ResourceConfig(num_gpus=1),
    )
"""

from __future__ import annotations

import base64
import io
import random
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.diffusion import DiffusionRequest, DiffusionResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_DTYPE_MAP = {
    "float32": None,  # resolved at load() time
    "float16": None,
    "bfloat16": None,
}


@register_backend("flux")
class FluxBackend(ModelBackend):
    """ModelBackend for FLUX image generation (black-forest-labs/FLUX.1-*).

    Args:
        model_id: HuggingFace model ID.  Options:
            "black-forest-labs/FLUX.1-schnell"  — fast (1–4 steps), Apache 2.0
            "black-forest-labs/FLUX.1-dev"      — higher quality, non-commercial
        device: "cpu", "cuda", or "mps".
        torch_dtype: "bfloat16" (recommended on Ampere+), "float16", or "float32".
        enable_model_cpu_offload: If True, offload model components to CPU
            between uses to reduce peak VRAM at the cost of speed.  Useful on
            GPUs with <12 GB VRAM.
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        enable_model_cpu_offload: bool = False,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._torch_dtype_str = torch_dtype
        self._enable_cpu_offload = enable_model_cpu_offload
        self._pipeline: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.DIFFUSION

    def load(self) -> None:
        try:
            import torch  # ty: ignore[unresolved-import]
            from diffusers import FluxPipeline  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "diffusers and torch are required for the FLUX backend. "
                "Install with: pip install 'sheaf-serve[diffusion]'"
            ) from e

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self._torch_dtype_str, torch.bfloat16)

        self._pipeline = FluxPipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
        )

        if self._enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to(self._device)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, DiffusionRequest):
            raise TypeError(f"Expected DiffusionRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: DiffusionRequest) -> DiffusionResponse:
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import torch  # ty: ignore[unresolved-import]

        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        generator = torch.Generator(device=self._device).manual_seed(seed)

        result = self._pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )

        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        return DiffusionResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            image_b64=image_b64,
            height=image.height,
            width=image.width,
            seed=seed,
        )
