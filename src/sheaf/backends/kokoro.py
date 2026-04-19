"""Kokoro TTS backend.

Requires: pip install "sheaf-serve[kokoro]"
Model: hexgrad/Kokoro-82M (default)

Kokoro is a lightweight, high-quality TTS model with per-request voice selection
and speed control. The KPipeline yields (graphemes, phonemes, audio) tuples for
each sentence chunk; chunks are concatenated into a single WAV response.

Key characteristics:
- Output sample rate: 24000 Hz (fixed by the model).
- Voices: American English — af_heart (default), af_bella, af_sarah, af_nicole,
  af_sky, am_adam, am_michael; British English — bf_emma, bf_isabella, bm_george,
  bm_lewis. Voice set varies by lang_code.
- Speed: 0.5–2.0 (set via TTSRequest.speed). Normal = 1.0.
- lang_code controls the phonemisation pipeline: 'a' = American English (default),
  'b' = British English; other codes: 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'zh'.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from sheaf.api.audio import TTSRequest, TTSResponse
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends._audio_utils import encode_wav
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_SAMPLE_RATE = 24000
_DEFAULT_VOICE = "af_heart"


@register_backend("kokoro")
class KokoroBackend(ModelBackend):
    """ModelBackend for Kokoro TTS (hexgrad/Kokoro-82M).

    Args:
        model_name: Informational label / HuggingFace repo ID. Default:
            "hexgrad/Kokoro-82M". The KPipeline resolves the model automatically.
        lang_code: Phonemisation pipeline code. 'a' = American English (default),
            'b' = British English, 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'pt', 'zh'.
        device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        model_name: str = "hexgrad/Kokoro-82M",
        lang_code: str = "a",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._lang_code = lang_code
        self._device = device
        self._pipeline: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.TTS

    def load(self) -> None:
        try:
            from kokoro import KPipeline  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "kokoro is required for the Kokoro backend. "
                "Install it with: pip install 'sheaf-serve[kokoro]'"
            ) from e
        self._pipeline = KPipeline(lang_code=self._lang_code, device=self._device)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TTSRequest):
            raise TypeError(f"Expected TTSRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: TTSRequest) -> TTSResponse:
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        voice = request.voice_preset or _DEFAULT_VOICE
        chunks: list[np.ndarray] = []
        for _, _, audio in self._pipeline(
            request.text, voice=voice, speed=request.speed
        ):
            chunks.append(audio)

        audio_np = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        wav_bytes = encode_wav(audio_np, _SAMPLE_RATE)

        return TTSResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            audio_b64=base64.b64encode(wav_bytes).decode(),
            sample_rate=_SAMPLE_RATE,
        )
