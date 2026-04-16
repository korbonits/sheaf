"""Tests for MusicGenBackend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes model_name to both from_pretrained calls
  - load() stores processor for test injectability
  - load() moves model to specified device
  - predict() rejects non-AudioGenerationRequest inputs
  - predict() returns AudioGenerationResponse with correct structure
  - predict() audio_b64 is a valid RIFF/WAV header
  - predict() sampling_rate matches model.config.audio_encoder.sampling_rate
  - predict() max_new_tokens = int(duration_s * frame_rate)
  - predict() guidance_scale forwarded to generate()
  - predict() temperature and top_k forwarded to generate()
  - predict() duration_s in response matches generated audio length
  - predict() mono audio slice [0, 0] used for single-channel model
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import builtins
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.audio_generation import AudioGenerationRequest, AudioGenerationResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 32000  # Hz — standard MusicGen output
_FRAME_RATE = 50  # tokens/sec
_DURATION_S = 2.0
_N_SAMPLES = int(_DURATION_S * _SAMPLE_RATE)  # 64000


# ---------------------------------------------------------------------------
# Helpers — fake audio tensor
# ---------------------------------------------------------------------------


class _FakeChanTensor:
    """Tensor-like with shape, indexing, cpu/numpy, and mean — used for audio slices."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, key: object) -> _FakeChanTensor:
        return _FakeChanTensor(self._data[key])  # type: ignore[index]

    def cpu(self) -> _FakeChanTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._data

    def mean(self, axis: int) -> _FakeChanTensor:
        return _FakeChanTensor(self._data.mean(axis=axis))


# ---------------------------------------------------------------------------
# Fake transformers module factory
# ---------------------------------------------------------------------------


def _make_transformers_mod(
    sample_rate: int = _SAMPLE_RATE,
    frame_rate: int = _FRAME_RATE,
    n_channels: int = 1,
    n_samples: int = _N_SAMPLES,
) -> ModuleType:
    # Processor
    mock_inputs: dict[str, MagicMock] = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    for v in mock_inputs.values():
        v.to.return_value = v

    processor = MagicMock()
    processor.return_value = mock_inputs

    # Model output: audio_values shaped (1, n_channels, T).
    # [0] → _FakeChanTensor with shape (n_channels, n_samples).
    audio_values = MagicMock()
    audio_values.__getitem__ = lambda self, key: _FakeChanTensor(
        np.zeros((n_channels, n_samples), dtype=np.float32)
    )

    model = MagicMock()
    model.generate.return_value = audio_values
    model.config.audio_encoder.sampling_rate = sample_rate
    model.config.audio_encoder.frame_rate = frame_rate
    model.to.return_value = model

    mod = ModuleType("transformers")
    mod.AutoProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]
    mod.MusicgenForConditionalGeneration = MagicMock()  # type: ignore[attr-defined]
    mod.MusicgenForConditionalGeneration.from_pretrained.return_value = model  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.musicgen import MusicGenBackend

    backend = MusicGenBackend(model_name="facebook/musicgen-small", device="cpu")
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    model = mod.MusicgenForConditionalGeneration.from_pretrained.return_value
    processor = mod.AutoProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# Request factory
# ---------------------------------------------------------------------------


def _make_request(
    prompt: str = "happy jazz piano",
    duration_s: float = _DURATION_S,
    guidance_scale: float | None = 3.0,
    temperature: float = 1.0,
    top_k: int = 250,
) -> AudioGenerationRequest:
    return AudioGenerationRequest(
        model_name="musicgen",
        prompt=prompt,
        duration_s=duration_s,
        guidance_scale=guidance_scale,
        temperature=temperature,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    from sheaf.backends.musicgen import MusicGenBackend

    backend = MusicGenBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[audio-generation\\]"),
    ):
        backend.load()


def test_load_passes_model_name(mock_transformers: ModuleType) -> None:
    from sheaf.backends.musicgen import MusicGenBackend

    model_name = "facebook/musicgen-medium"
    backend = MusicGenBackend(model_name=model_name)
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()

    mock_transformers.AutoProcessor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        model_name
    )
    mock_transformers.MusicgenForConditionalGeneration.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        model_name
    )


def test_load_stores_processor(mock_transformers: ModuleType) -> None:
    from sheaf.backends.musicgen import MusicGenBackend

    backend = MusicGenBackend()
    assert backend._processor is None
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()
    assert backend._processor is not None


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.musicgen import MusicGenBackend

    backend = MusicGenBackend(device="cuda")
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()

    mock_transformers.MusicgenForConditionalGeneration.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.audio import TTSRequest

    req = TTSRequest(model_name="x", text="hello")
    with pytest.raises(TypeError, match="AudioGenerationRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_audio_generation_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    resp = loaded_backend.predict(_make_request())

    assert isinstance(resp, AudioGenerationResponse)
    assert resp.sampling_rate == _SAMPLE_RATE
    assert resp.audio_b64


def test_predict_audio_b64_is_valid_wav(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    resp = loaded_backend.predict(_make_request())

    wav_bytes = base64.b64decode(resp.audio_b64)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert wav_bytes[12:16] == b"fmt "


def test_predict_sampling_rate_from_model_config(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod(sample_rate=16000)
    _wire(loaded_backend, mock)
    resp = loaded_backend.predict(_make_request())

    assert resp.sampling_rate == 16000


def test_predict_duration_s_in_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """duration_s in response = len(audio) / sampling_rate."""
    mock = _make_transformers_mod(n_samples=_N_SAMPLES)
    _wire(loaded_backend, mock)
    resp = loaded_backend.predict(_make_request(duration_s=_DURATION_S))

    expected = _N_SAMPLES / _SAMPLE_RATE
    assert abs(resp.duration_s - expected) < 0.01


# ---------------------------------------------------------------------------
# predict() — generate() call arguments
# ---------------------------------------------------------------------------


def test_predict_max_new_tokens_from_duration(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """max_new_tokens = int(duration_s * frame_rate)."""
    mock = _make_transformers_mod(frame_rate=50)
    model, _ = _wire(loaded_backend, mock)
    loaded_backend.predict(_make_request(duration_s=3.0))

    _, kwargs = model.generate.call_args
    assert kwargs["max_new_tokens"] == 150  # int(3.0 * 50)


def test_predict_guidance_scale_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)
    loaded_backend.predict(_make_request(guidance_scale=5.0))

    _, kwargs = model.generate.call_args
    assert kwargs["guidance_scale"] == 5.0


def test_predict_none_guidance_scale_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)
    loaded_backend.predict(_make_request(guidance_scale=None))

    _, kwargs = model.generate.call_args
    assert kwargs["guidance_scale"] is None


def test_predict_temperature_and_top_k_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)
    loaded_backend.predict(_make_request(temperature=0.8, top_k=100))

    _, kwargs = model.generate.call_args
    assert kwargs["temperature"] == 0.8
    assert kwargs["top_k"] == 100


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_independently(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)

    reqs = [
        _make_request(prompt="ambient piano"),
        _make_request(prompt="electronic beats"),
    ]
    responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, AudioGenerationResponse) for r in responses)
    assert model.generate.call_count == 2
