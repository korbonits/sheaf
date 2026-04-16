"""Bark TTS backend via HuggingFace transformers.

Requires: pip install "sheaf-serve[tts]"
Models: "suno/bark-small" (default), "suno/bark"

Bark is a transformer-based text-to-speech model by Suno AI. It generates
realistic audio from text, optionally conditioned on a speaker voice preset.
The HuggingFace transformers implementation is used here (actively maintained)
rather than the original suno-ai/bark package.

Key characteristics:
- Output sample rate: 24000 Hz (from model.generation_config.sample_rate).
- Voice presets: multilingual speaker presets, e.g. "v2/en_speaker_6",
  "v2/fr_speaker_1". Full list at https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683
- No torch version pin — works with the transformers-installed torch.
- CPU inference is slow (~30s for a short sentence); GPU strongly recommended.
"""

from __future__ import annotations

import base64
from typing import Any

from sheaf.api.audio import TTSRequest, TTSResponse
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends._audio_utils import encode_wav
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("bark")
class BarkBackend(ModelBackend):
    """ModelBackend for Bark TTS (suno/bark via HuggingFace transformers).

    Args:
        model_name: HuggingFace model ID. Options:
            "suno/bark-small" — faster, lower quality (~1.2B params)
            "suno/bark"       — higher quality (~9B params)
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "suno/bark-small",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.TTS

    def load(self) -> None:
        try:
            from transformers import (  # ty: ignore[unresolved-import]
                AutoProcessor,
                BarkModel,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the Bark backend. "
                "Install it with: pip install 'sheaf-serve[tts]'"
            ) from e
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = BarkModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TTSRequest):
            raise TypeError(f"Expected TTSRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: TTSRequest) -> TTSResponse:
        if self._model is None or self._processor is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        inputs = self._processor(
            request.text,
            voice_preset=request.voice_preset,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        audio_array = self._model.generate(**inputs)
        audio_np = audio_array.cpu().numpy().squeeze()
        sample_rate: int = self._model.generation_config.sample_rate

        wav_bytes = encode_wav(audio_np, sample_rate)

        return TTSResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            audio_b64=base64.b64encode(wav_bytes).decode(),
            sample_rate=sample_rate,
        )
