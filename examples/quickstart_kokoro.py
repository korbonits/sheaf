"""Kokoro text-to-speech quickstart.

Requirements:
    pip install "sheaf-serve[kokoro]"

Usage:
    python examples/quickstart_kokoro.py

Demonstrates:
  - Kokoro pipeline for high-quality TTS with voice selection
  - Default voice ("af_heart") and American English male ("am_michael")
  - Speed control (0.8×, 1.0×, 1.2×)
  - Saving output as WAV files
"""

from __future__ import annotations

import base64
import struct

from sheaf.api.audio import TTSRequest
from sheaf.backends.kokoro import KokoroBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_wav(audio_b64: str, path: str) -> None:
    """Decode base64 WAV bytes and write to disk."""
    wav_bytes = base64.b64decode(audio_b64)
    with open(path, "wb") as f:
        f.write(wav_bytes)
    # Parse RIFF header to report duration
    data_size = struct.unpack_from("<I", wav_bytes, 40)[0]
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    n_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits = struct.unpack_from("<H", wav_bytes, 34)[0]
    n_samples = data_size // (n_channels * bits // 8)
    duration_s = n_samples / sample_rate
    print(f"  Saved {path}  ({duration_s:.2f}s, {sample_rate} Hz)")


# ---------------------------------------------------------------------------
# Load backend
# ---------------------------------------------------------------------------

print("Loading Kokoro (downloads ~360 MB on first run)...")

backend = KokoroBackend(lang_code="a", device="cpu")  # "a" = American English
backend.load()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Default voice — "af_heart"
# ---------------------------------------------------------------------------

print('--- Default voice: "af_heart" ---')

req = TTSRequest(
    model_name="kokoro",
    text="Sheaf is a unified serving layer for non-text foundation models.",
)
resp = backend.predict(req)
save_wav(resp.audio_b64, "/tmp/kokoro_af_heart.wav")

# ---------------------------------------------------------------------------
# Different voice — "am_michael"
# ---------------------------------------------------------------------------

print('\n--- American male voice: "am_michael" ---')

req = TTSRequest(
    model_name="kokoro",
    text="Every model type gets a typed request and response contract.",
    voice_preset="am_michael",
)
resp = backend.predict(req)
save_wav(resp.audio_b64, "/tmp/kokoro_am_michael.wav")

# ---------------------------------------------------------------------------
# Speed variations
# ---------------------------------------------------------------------------

print("\n--- Speed control (default voice) ---")

for speed in [0.8, 1.0, 1.2]:
    req = TTSRequest(
        model_name="kokoro",
        text="Batching, caching, and scheduling are optimized per model type.",
        speed=speed,
    )
    resp = backend.predict(req)
    path = f"/tmp/kokoro_speed_{str(speed).replace('.', '_')}.wav"
    save_wav(resp.audio_b64, path)

print("\nDone.")
