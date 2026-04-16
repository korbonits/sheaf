"""Audio transcription quickstart using the Whisper backend.

Requirements:
    pip install "sheaf-serve[audio]"

Usage:
    python examples/quickstart_audio.py [path/to/audio.wav]

If no path is given, uses examples/sample.wav (included in the repo).
Any format that ffmpeg can decode is supported (wav, mp3, m4a, flac, ogg).
WAV files work without ffmpeg installed.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from sheaf.api.audio import AudioRequest
from sheaf.backends.whisper import WhisperBackend

audio_path = (
    Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample.wav"
)

print(f"Audio file : {audio_path}")
print(f"File size  : {audio_path.stat().st_size / 1024:.1f} KB")

audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

backend = WhisperBackend(model_size="turbo", device="cpu")
print("\nLoading model (downloads ~1.5 GB on first run)...")
backend.load()
print("Model loaded.")

# ------------------------------------------------------------------
# Basic transcription
# ------------------------------------------------------------------

req = AudioRequest(model_name="whisper", audio_b64=audio_b64)
resp = backend.predict(req)

print("\n--- Transcription ---")
print(resp.text.strip())
print(f"\nLanguage : {resp.language}")
print(f"Duration : {resp.duration:.2f}s")
print(f"Segments : {len(resp.segments)}")

# ------------------------------------------------------------------
# Word-level timestamps
# ------------------------------------------------------------------

req_words = AudioRequest(
    model_name="whisper",
    audio_b64=audio_b64,
    word_timestamps=True,
)
resp_words = backend.predict(req_words)

print("\n--- Word timestamps ---")
for seg in resp_words.segments:
    if seg.words:
        for w in seg.words:
            print(
                f"  [{w.start:5.2f}s – {w.end:5.2f}s]"
                f"  {w.word!r:20s}  p={w.probability:.2f}"
            )

# ------------------------------------------------------------------
# Translation to English (pass a non-English file to see the effect)
# ------------------------------------------------------------------

req_translate = AudioRequest(
    model_name="whisper",
    audio_b64=audio_b64,
    task="translate",
)
resp_translate = backend.predict(req_translate)
print("\n--- Translation ---")
print(resp_translate.text.strip())
