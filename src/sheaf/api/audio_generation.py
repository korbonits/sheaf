"""API contract for audio generation models (MusicGen, etc.)."""

from __future__ import annotations

from typing import Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class AudioGenerationRequest(BaseRequest):
    """Request contract for text-conditioned audio/music generation.

    Args:
        prompt: Text description of the audio to generate
            (e.g. "happy jazz with piano and drums").
        duration_s: Target duration in seconds. Converted to
            max_new_tokens via model.config.audio_encoder.frame_rate
            (50 tokens/sec for MusicGen).
        guidance_scale: Classifier-free guidance scale. Higher values
            steer generation closer to the prompt at the cost of diversity.
            Typical range: 1.0–10.0. None disables CFG.
        temperature: Sampling temperature. Higher values increase randomness.
        top_k: Top-k nucleus sampling parameter.
    """

    model_type: Literal[ModelType.AUDIO_GENERATION] = ModelType.AUDIO_GENERATION

    prompt: str
    duration_s: float = 5.0
    guidance_scale: float | None = 3.0
    temperature: float = 1.0
    top_k: int = 250


class AudioGenerationResponse(BaseResponse):
    """Response contract for audio generation."""

    model_type: Literal[ModelType.AUDIO_GENERATION] = ModelType.AUDIO_GENERATION

    # Base64-encoded 16-bit PCM WAV
    audio_b64: str

    # Sample rate in Hz (32000 for MusicGen)
    sampling_rate: int

    # Actual generated duration in seconds
    duration_s: float
