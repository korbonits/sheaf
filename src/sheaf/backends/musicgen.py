"""MusicGen audio generation backend via HuggingFace transformers.

Requires: pip install "sheaf-serve[audio-generation]"
Models: "facebook/musicgen-small" (default), "facebook/musicgen-medium",
        "facebook/musicgen-large", "facebook/musicgen-stereo-small", etc.

MusicGen is Meta's text-conditioned music generation model. It uses an
EnCodec audio codec and a transformer language model conditioned on T5
text embeddings. The audio codec runs at 50 tokens/sec, so duration in
seconds maps directly to max_new_tokens via
    max_new_tokens = int(duration_s * model.config.audio_encoder.frame_rate)

Key characteristics:
- Output sample rate: 32000 Hz (model.config.audio_encoder.sampling_rate).
- Frame rate: 50 tokens/sec (model.config.audio_encoder.frame_rate).
- Output shape: (1, 1, T) float32 — slice [0, 0] for 1D mono array.
- Stereo models produce (1, 2, T) — slice [0] for 2-channel array; mixed
  to mono for encoding here.
"""

from __future__ import annotations

import base64
from typing import Any

from sheaf.api.audio_generation import AudioGenerationRequest, AudioGenerationResponse
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends._audio_utils import encode_wav
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("musicgen")
class MusicGenBackend(ModelBackend):
    """ModelBackend for MusicGen (facebook/musicgen-* via HuggingFace transformers).

    Args:
        model_name: HuggingFace model ID. Options include:
            "facebook/musicgen-small"  — ~300M params, fastest
            "facebook/musicgen-medium" — ~1.5B params
            "facebook/musicgen-large"  — ~3.3B params
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "facebook/musicgen-small",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.AUDIO_GENERATION

    def load(self) -> None:
        try:
            from transformers import (  # ty: ignore[unresolved-import]
                AutoProcessor,
                MusicgenForConditionalGeneration,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the MusicGen backend. "
                "Install it with: pip install 'sheaf-serve[audio-generation]'"
            ) from e
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, AudioGenerationRequest):
            raise TypeError(f"Expected AudioGenerationRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: AudioGenerationRequest) -> AudioGenerationResponse:
        if self._model is None or self._processor is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        inputs = self._processor(
            text=[request.prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        frame_rate: int = self._model.config.audio_encoder.frame_rate
        max_new_tokens = int(request.duration_s * frame_rate)

        audio_values = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            guidance_scale=request.guidance_scale,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        # audio_values shape: (1, n_channels, T)
        # mono models: (1, 1, T) → [0, 0] gives 1D
        # stereo models: (1, 2, T) → [0] gives (2, T), mix to mono
        audio_tensor = audio_values[0]
        if audio_tensor.shape[0] > 1:
            # stereo → average channels to mono
            audio_np = audio_tensor.cpu().numpy().mean(axis=0)
        else:
            audio_np = audio_tensor[0].cpu().numpy()

        sampling_rate: int = self._model.config.audio_encoder.sampling_rate
        duration_s = len(audio_np) / sampling_rate

        wav_bytes = encode_wav(audio_np, sampling_rate)

        return AudioGenerationResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            audio_b64=base64.b64encode(wav_bytes).decode(),
            sampling_rate=sampling_rate,
            duration_s=duration_s,
        )
