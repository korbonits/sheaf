"""API contract for audio foundation models (Whisper, Bark, etc.)."""

from __future__ import annotations

import base64
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class WordTimestamp(BaseModel):
    """Word-level timestamp from Whisper when word_timestamps=True."""

    word: str
    start: float
    end: float
    probability: float


class AudioSegment(BaseModel):
    """A single transcription segment from Whisper."""

    id: int
    start: float  # seconds
    end: float  # seconds
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[WordTimestamp] | None = None


class AudioRequest(BaseRequest):
    """Request contract for audio transcription / translation.

    Audio is passed as base64-encoded bytes. Any format that ffmpeg can
    decode is accepted (wav, mp3, m4a, ogg, flac, etc.).

    Args:
        audio_b64: Base64-encoded audio file bytes.
        language: BCP-47 language code (e.g. "en", "fr") or None for
            auto-detection. English-only model variants (e.g. "tiny.en")
            ignore this field.
        task: "transcribe" returns text in the source language.
              "translate" transcribes and translates to English.
        word_timestamps: If True, each segment includes word-level
            start/end times and per-word probabilities.
        temperature: Sampling temperature for decoding. A tuple triggers
            fallback through successive values on failure.
        initial_prompt: Optional text prepended to the first window to
            condition the model (e.g. vocabulary hints, speaker context).
        vad_filter: Filter out silence before transcription using Silero VAD.
            Supported by faster-whisper; ignored by openai-whisper.
        beam_size: Beam search width for decoding. Higher = more accurate,
            slower. Supported by faster-whisper; ignored by openai-whisper.
    """

    model_type: Literal[ModelType.AUDIO] = ModelType.AUDIO

    audio_b64: str
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    word_timestamps: bool = False
    temperature: float = 0.0
    initial_prompt: str | None = None
    vad_filter: bool = False
    beam_size: int = 5

    @field_validator("audio_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("audio_b64 must be valid base64-encoded bytes") from e
        return v


class AudioResponse(BaseResponse):
    """Response contract for audio transcription / translation."""

    model_type: Literal[ModelType.AUDIO] = ModelType.AUDIO

    # Full concatenated transcription text
    text: str

    # Detected or specified language (BCP-47 code)
    language: str

    # Segment-level results — always populated
    segments: list[AudioSegment] = Field(default_factory=list)

    # Audio duration in seconds
    duration: float | None = None

    # Language detection confidence [0, 1] — populated by faster-whisper
    language_probability: float | None = None


class TTSRequest(BaseRequest):
    """Request contract for text-to-speech synthesis.

    Args:
        text: Input text to synthesize.
        voice_preset: Optional speaker voice preset. Bark: "v2/en_speaker_6" etc.
            Kokoro: "af_heart", "af_bella", "am_adam", "bf_emma", "bm_george", etc.
            None uses the backend's default voice.
        speed: Playback speed multiplier [0.5, 2.0]. Supported by Kokoro; ignored
            by Bark. Default 1.0 (normal speed).
    """

    model_type: Literal[ModelType.TTS] = ModelType.TTS

    text: str
    voice_preset: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class TTSResponse(BaseResponse):
    """Response contract for text-to-speech synthesis."""

    model_type: Literal[ModelType.TTS] = ModelType.TTS

    # Base64-encoded 16-bit PCM WAV
    audio_b64: str

    # Sample rate in Hz (typically 24000 for Bark)
    sample_rate: int
