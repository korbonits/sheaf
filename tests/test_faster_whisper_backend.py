"""Tests for FasterWhisperBackend — fully mocked, no faster-whisper install required.

Covers:
  - load() raises ImportError when faster-whisper is absent
  - load() succeeds with correct WhisperModel constructor args
  - predict() transcription — text, language, language_probability, segments, duration
  - predict() word_timestamps — WordTimestamp objects populated
  - predict() vad_filter and beam_size forwarded to transcribe()
  - predict() translation task
  - batch_predict() runs each request independently
  - segments generator is fully consumed (lazy evaluation)
"""

from __future__ import annotations

import base64
import math
import struct
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.audio import AudioRequest, AudioResponse, AudioSegment, WordTimestamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_b64(duration_s: float = 0.1, sample_rate: int = 16000) -> str:
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

_FAKE_SEGMENT = SimpleNamespace(
    id=0,
    seek=0,
    start=0.0,
    end=1.2,
    text=" Hello world.",
    tokens=[50364, 2425, 1002, 13],
    temperature=0.0,
    avg_logprob=-0.3,
    compression_ratio=1.1,
    no_speech_prob=0.02,
    words=None,
)

_FAKE_INFO = SimpleNamespace(
    language="en",
    language_probability=0.99,
    duration=1.2,
    duration_after_vad=1.2,
)

_FAKE_WORD_SEGMENT = SimpleNamespace(
    **{
        **vars(_FAKE_SEGMENT),
        "words": [
            SimpleNamespace(word=" Hello", start=0.0, end=0.6, probability=0.98),
            SimpleNamespace(word=" world.", start=0.6, end=1.2, probability=0.95),
        ],
    },
)


# ---------------------------------------------------------------------------
# Mock faster_whisper module fixture
# ---------------------------------------------------------------------------


def _make_faster_whisper_mod(segments=None, info=None) -> ModuleType:
    mod = ModuleType("faster_whisper")
    _segments = segments if segments is not None else [_FAKE_SEGMENT]
    _info = info if info is not None else _FAKE_INFO

    model = MagicMock()
    model.transcribe.return_value = (iter(_segments), _info)
    mod.WhisperModel = MagicMock(return_value=model)  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def mock_fw() -> ModuleType:
    return _make_faster_whisper_mod()


@pytest.fixture
def loaded_backend(mock_fw: ModuleType):
    from sheaf.backends.faster_whisper import FasterWhisperBackend

    backend = FasterWhisperBackend(
        model_size="turbo", device="cpu", compute_type="int8"
    )
    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_faster_whisper() -> None:
    from sheaf.backends.faster_whisper import FasterWhisperBackend

    backend = FasterWhisperBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "faster_whisper" not in k}

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "faster_whisper":
            raise ModuleNotFoundError("No module named 'faster_whisper'")
        return __import__(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[audio\\]"),
    ):
        backend.load()


def test_load_passes_correct_args(mock_fw: ModuleType) -> None:
    from sheaf.backends.faster_whisper import FasterWhisperBackend

    backend = FasterWhisperBackend(
        model_size="small",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root="/tmp/weights",
    )
    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        backend.load()

    mock_fw.WhisperModel.assert_called_once_with(  # type: ignore[attr-defined]
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root="/tmp/weights",
    )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_transcription(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(model_name="faster-whisper", audio_b64=_AUDIO_B64)
    mock = _make_faster_whisper_mod()
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, AudioResponse)
    assert resp.text == " Hello world."
    assert resp.language == "en"
    assert resp.language_probability == 0.99
    assert resp.duration == 1.2
    assert len(resp.segments) == 1
    seg = resp.segments[0]
    assert isinstance(seg, AudioSegment)
    assert seg.start == 0.0
    assert seg.end == 1.2
    assert seg.words is None


def test_predict_language_probability_populated(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """language_probability from TranscriptionInfo surfaces in AudioResponse."""
    req = AudioRequest(model_name="faster-whisper", audio_b64=_AUDIO_B64)
    mock = _make_faster_whisper_mod(
        info=SimpleNamespace(
            language="fr",
            language_probability=0.87,
            duration=1.2,
            duration_after_vad=1.2,
        )
    )
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        resp = loaded_backend.predict(req)

    assert resp.language == "fr"
    assert resp.language_probability == pytest.approx(0.87)


def test_predict_word_timestamps(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(
        model_name="faster-whisper",
        audio_b64=_AUDIO_B64,
        word_timestamps=True,
    )
    mock = _make_faster_whisper_mod(segments=[_FAKE_WORD_SEGMENT])
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        resp = loaded_backend.predict(req)

    words = resp.segments[0].words
    assert words is not None
    assert len(words) == 2
    assert isinstance(words[0], WordTimestamp)
    assert words[0].word == " Hello"
    assert words[0].probability == 0.98


def test_predict_vad_filter_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(
        model_name="faster-whisper",
        audio_b64=_AUDIO_B64,
        vad_filter=True,
        beam_size=3,
    )
    mock = _make_faster_whisper_mod()
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        loaded_backend.predict(req)

    _, kwargs = loaded_backend._model.transcribe.call_args
    assert kwargs["vad_filter"] is True
    assert kwargs["beam_size"] == 3


def test_predict_translation_task(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = AudioRequest(
        model_name="faster-whisper",
        audio_b64=_AUDIO_B64,
        task="translate",
        language="de",
    )
    mock = _make_faster_whisper_mod()
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        loaded_backend.predict(req)

    _, kwargs = loaded_backend._model.transcribe.call_args
    assert kwargs["task"] == "translate"
    assert kwargs["language"] == "de"


def test_segments_generator_consumed(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Generator must be fully consumed — partial iteration drops segments."""
    segments = [
        SimpleNamespace(**{**vars(_FAKE_SEGMENT), "id": i, "text": f" seg{i}"})
        for i in range(3)
    ]
    req = AudioRequest(model_name="faster-whisper", audio_b64=_AUDIO_B64)
    mock = _make_faster_whisper_mod(segments=segments)
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        resp = loaded_backend.predict(req)

    assert len(resp.segments) == 3
    assert resp.text == " seg0 seg1 seg2"


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    reqs = [
        AudioRequest(model_name="faster-whisper", audio_b64=_AUDIO_B64),
        AudioRequest(model_name="faster-whisper", audio_b64=_AUDIO_B64, language="es"),
    ]
    mock = _make_faster_whisper_mod()
    loaded_backend._model = mock.WhisperModel()

    with patch.dict(sys.modules, {"faster_whisper": mock}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert loaded_backend._model.transcribe.call_count == 2
