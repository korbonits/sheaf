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

import asyncio
import base64
import io
import random
from collections.abc import AsyncGenerator
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.diffusion import DiffusionRequest, DiffusionResponse
from sheaf.backends.base import ModelBackend
from sheaf.lora import LoRAAdapter
from sheaf.lora import parse_source as _parse_lora_source
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

    # ------------------------------------------------------------------
    # LoRA adapter multiplexing
    # ------------------------------------------------------------------

    def supports_lora(self) -> bool:
        return True

    def load_adapters(self, adapters: dict[str, LoRAAdapter]) -> None:
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")
        for name, adapter in adapters.items():
            path_or_repo, weight_name = _parse_lora_source(adapter.source)
            kwargs: dict[str, Any] = {"adapter_name": name}
            if weight_name is not None:
                kwargs["weight_name"] = weight_name
            self._pipeline.load_lora_weights(path_or_repo, **kwargs)

    def set_active_adapters(self, names: list[str], weights: list[float]) -> None:
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")
        if len(names) != len(weights):
            raise ValueError(
                f"names length ({len(names)}) must equal weights length "
                f"({len(weights)})"
            )
        # Empty names means "no LoRA for this sub-batch".  Diffusers'
        # set_adapters([], []) raises KeyError on the 'transformer' component
        # rather than disabling — disable_lora() is the supported off-switch.
        if not names:
            self._pipeline.disable_lora()
            return
        self._pipeline.enable_lora()
        self._pipeline.set_adapters(names, adapter_weights=weights)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, DiffusionRequest):
            raise TypeError(f"Expected DiffusionRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(
        self,
        request: DiffusionRequest,
        callback: Any = None,
    ) -> DiffusionResponse:
        """Run the FLUX pipeline synchronously.

        Args:
            request: The diffusion request.
            callback: Optional ``callback_on_step_end`` callable passed to the
                pipeline for streaming support.  Called as
                ``callback(pipe, step_idx, timestep, callback_kwargs)`` and must
                return ``callback_kwargs``.
        """
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import torch  # ty: ignore[unresolved-import]

        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        generator = torch.Generator(device=self._device).manual_seed(seed)

        extra: dict[str, Any] = {}
        if callback is not None:
            extra["callback_on_step_end"] = callback

        result = self._pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            **extra,
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

    async def stream_predict(
        self, request: BaseRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream inference steps as Server-Sent Events.

        Yields one ``{"type": "progress", ...}`` event per denoising step,
        then a final ``{"type": "result", "done": True, ...}`` event with the
        completed image.

        Uses a thread-safe ``queue.Queue`` to bridge the synchronous pipeline
        callback (running in a thread-pool executor) to this async generator.
        """
        from queue import Empty, Queue

        if not isinstance(request, DiffusionRequest):
            raise TypeError(f"Expected DiffusionRequest, got {type(request)}")

        q: Queue[dict[str, Any]] = Queue()
        num_steps = request.num_inference_steps

        def _step_callback(
            pipe: Any,
            step_idx: int,
            timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            q.put(
                {
                    "type": "progress",
                    "step": step_idx + 1,
                    "total_steps": num_steps,
                    "done": False,
                }
            )
            return callback_kwargs

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, lambda: self._run(request, _step_callback))

        # Yield progress events while the executor thread is running.
        while not future.done():
            try:
                yield q.get_nowait()
            except Empty:
                await asyncio.sleep(0.02)

        # Drain any events that arrived after the last done() check.
        while True:
            try:
                yield q.get_nowait()
            except Empty:
                break

        # Await the future to propagate any exception from the executor thread.
        response = await future
        yield {"type": "result", "done": True, **response.model_dump(mode="json")}
