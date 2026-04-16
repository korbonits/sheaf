"""Tests for WhisperBackend — fully mocked, no real openai-whisper install required.

Covers:
  - load() raises ImportError when openai-whisper is absent
  - load() succeeds when whisper is present (module stub)
  - predict() transcription — text, language, segments
  - predict() translation task
  - predict() word_timestamps — WordTimestamp objects populated
  - batch_predict() runs each request independently
  - AudioRequest base64 validation
"""

from __future__ import annotations

import base64
import math
import struct
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.audio import AudioRequest, AudioResponse, AudioSegment, WordTimestamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_b64(duration_s: float = 0.1, sample_rate: int = 16000) -> str:
    """Return a minimal valid WAV file as a base64 string."""
    num_samples = int(sample_rate * duration_s)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(num_samples)
    ]
    audio_data = struct.pack("<" + "h" * num_samples, *samples)
    data_size = len(audio_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return base64.b64encode(header + audio_data).decode()


_AUDIO_B64 = _make_wav_b64()

_FAKE_RESULT = {
    "text": " Hello world.",
    "language": "en",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.2,
            "text": " Hello world.",
            "tokens": [50364, 2425, 1002, 13],
            "temperature": 0.0,
            "avg_logprob": -0.3,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.02,
        }
    ],
}

_FAKE_RESULT_WITH_WORDS = {
    **_FAKE_RESULT,
    "segments": [
        {
            **_FAKE_RESULT["segments"][0],
            "words": [
                {"word": " Hello", "start": 0.0, "end": 0.6, "probability": 0.98},
                {"word": " world.", "start": 0.6, "end": 1.2, "probability": 0.95},
            ],
        }
    ],
}


# ---------------------------------------------------------------------------
# Mock whisper module fixture
# ---------------------------------------------------------------------------


def _make_whisper_mod(result: dict | None = None) -> ModuleType:
    """Fake openai-whisper module with a stub model."""
    mod = ModuleType("whisper")

    model = MagicMock()
    model.transcribe.return_value = result or _FAKE_RESULT

    mod.load_model = MagicMock(return_value=model)  # type: ignore[attr-defined]
    mod.available_models = MagicMock(  # type: ignore[attr-defined]
        return_value=["tiny", "base", "small", "medium", "large", "turbo"]
    )
    return mod


@pytest.fixture
def mock_whisper() -> ModuleType:
    return _make_whisper_mod()


@pytest.fixture
def loaded_backend(mock_whisper: ModuleType):
    from sheaf.backends.whisper import WhisperBackend

    backend = WhisperBackend(model_size="turbo", device="cpu")
    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_whisper() -> None:
    """load() raises ImportError when openai-whisper is not installed."""
    from sheaf.backends.whisper import WhisperBackend

    backend = WhisperBackend()
    mods_without = {k: v for k, v in sys.modules.items() if k != "whisper"}

    def _raise_on_whisper(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "whisper":
            raise ModuleNotFoundError("No module named 'whisper'")
        return __import__(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise_on_whisper),
        pytest.raises(ImportError, match="sheaf-serve\\[audio\\]"),
    ):
        backend.load()


def test_load_succeeds(mock_whisper: ModuleType) -> None:
    from sheaf.backends.whisper import WhisperBackend

    backend = WhisperBackend(model_size="small", device="cpu")
    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        backend.load()
    assert backend._model is not None
    mock_whisper.load_model.assert_called_once_with(
        "small", device="cpu", download_root=None
    )


# ---------------------------------------------------------------------------
# AudioRequest validation
# ---------------------------------------------------------------------------


def test_audio_request_invalid_base64() -> None:
    with pytest.raises(Exception, match="base64"):
        AudioRequest(
            model_name="whisper",
            audio_b64="not-valid-base64!!!",
        )


def test_audio_request_valid() -> None:
    req = AudioRequest(model_name="whisper", audio_b64=_AUDIO_B64)
    assert req.task == "transcribe"
    assert req.language is None
    assert req.word_timestamps is False


# ---------------------------------------------------------------------------
# predict() — transcription
# ---------------------------------------------------------------------------


def test_predict_transcription(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(model_name="whisper", audio_b64=_AUDIO_B64)

    with patch.dict(sys.modules, {"whisper": _make_whisper_mod()}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, AudioResponse)
    assert resp.text == " Hello world."
    assert resp.language == "en"
    assert len(resp.segments) == 1
    seg = resp.segments[0]
    assert isinstance(seg, AudioSegment)
    assert seg.start == 0.0
    assert seg.end == 1.2
    assert seg.text == " Hello world."
    assert seg.words is None
    assert resp.duration == 1.2


def test_predict_translation_task(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(
        model_name="whisper",
        audio_b64=_AUDIO_B64,
        task="translate",
        language="fr",
    )
    mock = _make_whisper_mod()
    with patch.dict(sys.modules, {"whisper": mock}):
        loaded_backend.predict(req)

    _, kwargs = loaded_backend._model.transcribe.call_args
    assert kwargs["task"] == "translate"
    assert kwargs["language"] == "fr"


# ---------------------------------------------------------------------------
# predict() — word timestamps
# ---------------------------------------------------------------------------


def test_predict_word_timestamps(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(
        model_name="whisper",
        audio_b64=_AUDIO_B64,
        word_timestamps=True,
    )
    mock = _make_whisper_mod(result=_FAKE_RESULT_WITH_WORDS)
    loaded_backend._model = mock.load_model()

    with patch.dict(sys.modules, {"whisper": mock}):
        resp = loaded_backend.predict(req)

    assert resp.segments[0].words is not None
    assert len(resp.segments[0].words) == 2
    w = resp.segments[0].words[0]
    assert isinstance(w, WordTimestamp)
    assert w.word == " Hello"
    assert w.start == 0.0
    assert w.probability == 0.98


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    reqs = [
        AudioRequest(model_name="whisper", audio_b64=_AUDIO_B64),
        AudioRequest(model_name="whisper", audio_b64=_AUDIO_B64, language="es"),
    ]
    mock = _make_whisper_mod()
    loaded_backend._model = mock.load_model()

    with patch.dict(sys.modules, {"whisper": mock}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, AudioResponse) for r in responses)
    # Each request is run independently
    assert loaded_backend._model.transcribe.call_count == 2
