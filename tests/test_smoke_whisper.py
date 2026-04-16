"""End-to-end smoke test for the Whisper backend against real audio.

Downloads the turbo model (~1.5 GB) on first run and transcribes
examples/sample.wav (a 4.8s macOS TTS clip included in the repo).

Run explicitly:
    SHEAF_SMOKE_TEST=1 .venv/bin/pytest tests/test_smoke_whisper.py -v -s

Skipped in normal CI — requires the model weights and takes ~90s on CPU.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from sheaf.api.audio import AudioRequest, AudioResponse, AudioSegment, WordTimestamp
from sheaf.backends.whisper import WhisperBackend

pytestmark = pytest.mark.skipif(
    not os.environ.get("SHEAF_SMOKE_TEST"),
    reason="Set SHEAF_SMOKE_TEST=1 to run Whisper smoke tests",
)

_SAMPLE_WAV = Path(__file__).parent.parent / "examples" / "sample.wav"
_EXPECTED_WORDS = {"quick", "brown", "fox", "lazy", "dog", "foundation", "models"}


@pytest.fixture(scope="module")
def backend() -> WhisperBackend:
    b = WhisperBackend(model_size="turbo", device="cpu")
    b.load()
    return b


@pytest.fixture(scope="module")
def audio_b64() -> str:
    return base64.b64encode(_SAMPLE_WAV.read_bytes()).decode()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sample_wav_exists() -> None:
    assert _SAMPLE_WAV.exists(), f"Sample WAV not found at {_SAMPLE_WAV}"


def test_transcription_returns_response(
    backend: WhisperBackend, audio_b64: str
) -> None:
    req = AudioRequest(model_name="whisper", audio_b64=audio_b64)
    resp = backend.predict(req)

    assert isinstance(resp, AudioResponse)
    assert resp.language == "en"
    assert resp.text.strip() != ""
    assert resp.duration is not None and resp.duration > 0
    assert len(resp.segments) > 0


def test_transcription_contains_expected_words(
    backend: WhisperBackend, audio_b64: str
) -> None:
    req = AudioRequest(model_name="whisper", audio_b64=audio_b64)
    resp = backend.predict(req)

    words_in_transcript = set(resp.text.lower().split())
    matched = _EXPECTED_WORDS & words_in_transcript
    assert len(matched) >= 5, (
        f"Expected at least 5 of {_EXPECTED_WORDS} in transcript, "
        f"got {matched!r}. Full text: {resp.text!r}"
    )


def test_word_timestamps(backend: WhisperBackend, audio_b64: str) -> None:
    req = AudioRequest(
        model_name="whisper",
        audio_b64=audio_b64,
        word_timestamps=True,
    )
    resp = backend.predict(req)

    assert all(isinstance(seg, AudioSegment) for seg in resp.segments)
    words = [w for seg in resp.segments if seg.words for w in seg.words]
    assert len(words) > 0
    for w in words:
        assert isinstance(w, WordTimestamp)
        assert w.start >= 0
        assert w.end >= w.start
        assert 0.0 <= w.probability <= 1.0


def test_translation_task(backend: WhisperBackend, audio_b64: str) -> None:
    """Translation of English audio should still return English text."""
    req = AudioRequest(
        model_name="whisper",
        audio_b64=audio_b64,
        task="translate",
    )
    resp = backend.predict(req)
    assert resp.text.strip() != ""


def test_segments_are_chronological(backend: WhisperBackend, audio_b64: str) -> None:
    req = AudioRequest(model_name="whisper", audio_b64=audio_b64)
    resp = backend.predict(req)

    for i in range(1, len(resp.segments)):
        assert resp.segments[i].start >= resp.segments[i - 1].start, (
            f"Segment {i} starts before segment {i - 1}"
        )


def test_batch_predict(backend: WhisperBackend, audio_b64: str) -> None:
    reqs = [
        AudioRequest(model_name="whisper", audio_b64=audio_b64),
        AudioRequest(model_name="whisper", audio_b64=audio_b64, language="en"),
    ]
    responses = backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, AudioResponse) for r in responses)
    assert responses[0].text == responses[1].text
