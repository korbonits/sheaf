"""Whisper backend for audio transcription and translation.

Requires: pip install "sheaf-serve[audio]"
Model family: openai/whisper — tiny, base, small, medium, large-v3, turbo
"""

from __future__ import annotations

import base64
from typing import Any

from sheaf.api.audio import AudioRequest, AudioResponse, AudioSegment, WordTimestamp
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends._audio_utils import decode_audio
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("whisper")
class WhisperBackend(ModelBackend):
    """ModelBackend for OpenAI Whisper (transcription + translation).

    Audio is accepted as base64-encoded bytes in the request; the backend
    decodes to a temporary file and passes the path to whisper.transcribe().
    Any format that ffmpeg can decode is accepted (wav, mp3, m4a, flac, etc.).

    Args:
        model_size: Whisper model variant. Options:
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large-v1", "large-v2", "large-v3", "large",
            "large-v3-turbo", "turbo"
        device: "cpu", "cuda", or "auto"
        download_root: Optional local directory for model weight cache.
    """

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cpu",
        download_root: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._download_root = download_root
        self._model: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.AUDIO

    def load(self) -> None:
        try:
            import whisper  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "openai-whisper is required for the Whisper backend. "
                "Install it with: pip install 'sheaf-serve[audio]'"
            ) from e
        self._model = whisper.load_model(
            self._model_size,
            device=self._device,
            download_root=self._download_root,
        )

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, AudioRequest):
            raise TypeError(f"Expected AudioRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        # Whisper's bottleneck is per-file I/O and model inference.
        # Each request has independent audio, so we run sequentially.
        # Future: batched mel spectrogram extraction.
        return [self.predict(r) for r in requests]

    def _run(self, request: AudioRequest) -> AudioResponse:
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        audio_bytes = base64.b64decode(request.audio_b64)
        audio_input = decode_audio(audio_bytes)

        result = self._model.transcribe(
            audio_input,
            language=request.language,
            task=request.task,
            word_timestamps=request.word_timestamps,
            temperature=request.temperature,
            initial_prompt=request.initial_prompt,
            verbose=False,
        )

        return self._build_response(request, result)

    def _build_response(
        self, request: AudioRequest, result: dict[str, Any]
    ) -> AudioResponse:
        segments = [
            AudioSegment(
                id=seg["id"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                avg_logprob=seg["avg_logprob"],
                no_speech_prob=seg["no_speech_prob"],
                words=(
                    [
                        WordTimestamp(
                            word=w["word"],
                            start=w["start"],
                            end=w["end"],
                            probability=w["probability"],
                        )
                        for w in seg.get("words", [])
                    ]
                    if request.word_timestamps
                    else None
                ),
            )
            for seg in result.get("segments", [])
        ]

        # Compute duration from the last segment end if available.
        duration = segments[-1].end if segments else None

        return AudioResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            text=result["text"],
            language=result["language"],
            segments=segments,
            duration=duration,
        )
