"""Tests for KokoroBackend — fully mocked, no real kokoro install required.

Covers:
  - load() raises ImportError when kokoro is absent
  - load() passes correct lang_code and device to KPipeline
  - model_type returns ModelType.TTS
  - predict() returns a TTSResponse with audio_b64 and sample_rate=24000
  - predict() output is a valid 16-bit PCM WAV (RIFF header check)
  - predict() uses voice_preset when provided
  - predict() defaults to "af_heart" when voice_preset is None
  - predict() forwards speed to the pipeline
  - predict() concatenates multiple audio chunks from the generator
  - predict() handles an empty generator (yields no chunks)
  - predict() raises TypeError on wrong request type
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.audio import TTSRequest, TTSResponse
from sheaf.api.base import ModelType
from sheaf.api.tabular import TabularRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK = np.ones(4800, dtype=np.float32) * 0.1  # 0.2 s at 24 kHz


def _make_kokoro_mod(chunks: list[np.ndarray] | None = None) -> ModuleType:
    """Build a minimal fake `kokoro` module."""
    _chunks = chunks if chunks is not None else [_CHUNK]

    mod = ModuleType("kokoro")
    pipeline_instance = MagicMock()
    # Calling the pipeline returns an iterable of (graphemes, phonemes, audio) tuples.
    pipeline_instance.return_value = iter([("g", "p", chunk) for chunk in _chunks])
    pipeline_cls = MagicMock(return_value=pipeline_instance)
    mod.KPipeline = pipeline_cls  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def mock_kokoro() -> ModuleType:
    return _make_kokoro_mod()


@pytest.fixture
def loaded_backend(mock_kokoro: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.kokoro import KokoroBackend

    backend = KokoroBackend(
        model_name="hexgrad/Kokoro-82M", lang_code="a", device="cpu"
    )
    with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_kokoro() -> None:
    from sheaf.backends.kokoro import KokoroBackend

    backend = KokoroBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "kokoro" not in k}

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "kokoro":
            raise ModuleNotFoundError("No module named 'kokoro'")
        return __import__(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[kokoro\\]"),
    ):
        backend.load()


def test_load_passes_lang_code_and_device(mock_kokoro: ModuleType) -> None:
    from sheaf.backends.kokoro import KokoroBackend

    backend = KokoroBackend(lang_code="b", device="cuda")
    with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
        backend.load()

    mock_kokoro.KPipeline.assert_called_once_with(lang_code="b", device="cuda")  # type: ignore[attr-defined]


def test_model_type() -> None:
    from sheaf.backends.kokoro import KokoroBackend

    assert KokoroBackend().model_type == ModelType.TTS


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_returns_tts_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="kokoro", text="Hello world.")
    mock = _make_kokoro_mod()
    loaded_backend._pipeline = mock.KPipeline.return_value

    resp = loaded_backend.predict(req)

    assert isinstance(resp, TTSResponse)
    assert resp.sample_rate == 24000
    assert resp.audio_b64


def test_predict_audio_b64_is_valid_wav(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="kokoro", text="Test.")
    mock = _make_kokoro_mod()
    loaded_backend._pipeline = mock.KPipeline.return_value

    resp = loaded_backend.predict(req)

    wav_bytes = base64.b64decode(resp.audio_b64)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert wav_bytes[12:16] == b"fmt "


def test_predict_uses_voice_preset(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="kokoro", text="Hi.", voice_preset="af_bella")
    mock = _make_kokoro_mod()
    pipeline = mock.KPipeline.return_value
    loaded_backend._pipeline = pipeline

    loaded_backend.predict(req)

    pipeline.assert_called_once_with("Hi.", voice="af_bella", speed=1.0)


def test_predict_defaults_voice_to_af_heart(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="kokoro", text="Hi.")
    mock = _make_kokoro_mod()
    pipeline = mock.KPipeline.return_value
    loaded_backend._pipeline = pipeline

    loaded_backend.predict(req)

    _, kwargs = pipeline.call_args
    assert kwargs["voice"] == "af_heart"


def test_predict_forwards_speed(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="kokoro", text="Fast.", speed=1.5)
    mock = _make_kokoro_mod()
    pipeline = mock.KPipeline.return_value
    loaded_backend._pipeline = pipeline

    loaded_backend.predict(req)

    _, kwargs = pipeline.call_args
    assert kwargs["speed"] == pytest.approx(1.5)


def test_predict_concatenates_multiple_chunks(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    chunk_a = np.zeros(4800, dtype=np.float32)
    chunk_b = np.ones(4800, dtype=np.float32) * 0.5
    mock = _make_kokoro_mod(chunks=[chunk_a, chunk_b])
    loaded_backend._pipeline = mock.KPipeline.return_value

    req = TTSRequest(model_name="kokoro", text="Two chunks.")
    resp = loaded_backend.predict(req)

    wav_bytes = base64.b64decode(resp.audio_b64)
    # WAV header is 44 bytes; 9600 samples × 2 bytes/sample = 19200 bytes of audio
    assert len(wav_bytes) == 44 + 9600 * 2


def test_predict_empty_generator(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_kokoro_mod(chunks=[])
    loaded_backend._pipeline = mock.KPipeline.return_value

    req = TTSRequest(model_name="kokoro", text="")
    resp = loaded_backend.predict(req)

    assert isinstance(resp, TTSResponse)
    wav_bytes = base64.b64decode(resp.audio_b64)
    # 0 samples → WAV header only (44 bytes)
    assert len(wav_bytes) == 44


def test_predict_raises_on_wrong_request_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    bad_req = TabularRequest(
        model_name="kokoro",
        context_X=[[1.0, 2.0]],
        context_y=[0],
        query_X=[[1.0, 2.0]],
        task="classification",
    )
    with pytest.raises(TypeError, match="TTSRequest"):
        loaded_backend.predict(bad_req)


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_each_request_independently(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    reqs = [
        TTSRequest(model_name="kokoro", text="Hello."),
        TTSRequest(
            model_name="kokoro", text="World.", voice_preset="am_adam", speed=0.8
        ),
    ]
    # Give the pipeline a fresh generator per call via side_effect.
    pipeline = MagicMock()
    pipeline.side_effect = [
        iter([("g", "p", _CHUNK)]),
        iter([("g", "p", _CHUNK)]),
    ]
    loaded_backend._pipeline = pipeline

    responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, TTSResponse) for r in responses)
    assert pipeline.call_count == 2
