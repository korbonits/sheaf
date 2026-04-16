"""faster-whisper backend for audio transcription and translation.

Requires: pip install "sheaf-serve[audio]"
Model family: same checkpoints as openai/whisper, run via CTranslate2.

Key differences from WhisperBackend:
- Uses CTranslate2 (not PyTorch) at inference time — no torch runtime dep.
- transcribe() returns a lazy generator; segments are consumed here.
- Built-in VAD filtering via Silero VAD (vad_filter=True on AudioRequest).
- language_probability is available on TranscriptionInfo and surfaced
  in AudioResponse.
- compute_type controls quantisation: "int8" (fastest/CPU), "float32",
  "float16" / "int8_float16" (GPU).
"""

from __future__ import annotations

import base64
from typing import Any

from sheaf.api.audio import AudioRequest, AudioResponse, AudioSegment, WordTimestamp
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends._audio_utils import decode_audio
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("faster-whisper")
class FasterWhisperBackend(ModelBackend):
    """ModelBackend for faster-whisper (CTranslate2-based Whisper).

    Drop-in alternative to WhisperBackend with ~4x faster inference on CPU
    via CTranslate2 quantisation. Accepts the same AudioRequest contract.

    Args:
        model_size: Model variant — same names as openai/whisper plus
            distilled variants: "distil-large-v3", "distil-medium.en", etc.
        device: "cpu", "cuda", or "auto"
        compute_type: Quantisation type. Recommended:
            "int8" for CPU (fastest), "float16" or "int8_float16" for GPU,
            "float32" for maximum precision.
        cpu_threads: Number of threads for CPU inference. 0 = auto.
        download_root: Optional local directory for model weight cache.
    """

    def __init__(
        self,
        model_size: str = "turbo",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        download_root: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._download_root = download_root
        self._model: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.AUDIO

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "faster-whisper is required for the FasterWhisper backend. "
                "Install it with: pip install 'sheaf-serve[audio]'"
            ) from e
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
            download_root=self._download_root,
        )

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, AudioRequest):
            raise TypeError(f"Expected AudioRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: AudioRequest) -> AudioResponse:
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        audio_bytes = base64.b64decode(request.audio_b64)
        audio_input = decode_audio(audio_bytes)

        # transcribe() returns (segments_generator, info) — consume the generator.
        segments_gen, info = self._model.transcribe(
            audio_input,
            language=request.language,
            task=request.task,
            beam_size=request.beam_size,
            word_timestamps=request.word_timestamps,
            temperature=request.temperature,
            initial_prompt=request.initial_prompt,
            vad_filter=request.vad_filter,
        )

        segments = [
            AudioSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
                words=(
                    [
                        WordTimestamp(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                        for w in (seg.words or [])
                    ]
                    if request.word_timestamps
                    else None
                ),
            )
            for seg in segments_gen  # consuming the generator runs inference
        ]

        return AudioResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            text="".join(seg.text for seg in segments),
            language=info.language,
            language_probability=info.language_probability,
            segments=segments,
            duration=info.duration if info.duration > 0 else None,
        )
