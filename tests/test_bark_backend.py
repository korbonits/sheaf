"""Tests for BarkBackend — fully mocked, no real transformers/torch install required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes correct model_name to from_pretrained()
  - load() moves model to the specified device
  - predict() returns a TTSResponse with audio_b64 and sample_rate
  - predict() output is a valid 16-bit PCM WAV (RIFF header check)
  - predict() forwards voice_preset to the processor
  - predict() forwards None voice_preset correctly
  - predict() sample_rate matches model.generation_config.sample_rate
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_AUDIO = np.zeros(24000, dtype=np.float32)  # 1 s silence at 24 kHz


def _make_transformers_mod(
    audio_output: np.ndarray | None = None,
    sample_rate: int = 24000,
) -> ModuleType:
    mod = ModuleType("transformers")
    _audio = audio_output if audio_output is not None else _FAKE_AUDIO

    # Processor: __call__ returns a dict of mock tensors, each supporting .to()
    mock_inputs: dict[str, MagicMock] = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    for v in mock_inputs.values():
        v.to.return_value = v

    processor = MagicMock()
    processor.return_value = mock_inputs
    mod.AutoProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]

    # Model: generate() → tensor → .cpu().numpy().squeeze() → ndarray
    audio_tensor = MagicMock()
    audio_tensor.cpu.return_value.numpy.return_value.squeeze.return_value = _audio

    model = MagicMock()
    model.generate.return_value = audio_tensor
    model.generation_config.sample_rate = sample_rate
    model.to.return_value = model  # .to(device) returns the same mock

    mod.BarkModel = MagicMock()  # type: ignore[attr-defined]
    mod.BarkModel.from_pretrained.return_value = model  # type: ignore[attr-defined]

    return mod


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.bark import BarkBackend

    backend = BarkBackend(model_name="suno/bark-small", device="cpu")
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# Helpers to wire a fresh mock into an already-loaded backend
# ---------------------------------------------------------------------------


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    """Replace backend._model and _processor with fresh mock instances."""
    model = mod.BarkModel.from_pretrained.return_value
    processor = mod.AutoProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    from sheaf.backends.bark import BarkBackend

    backend = BarkBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return __import__(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[tts\\]"),
    ):
        backend.load()


def test_load_passes_correct_model_name(mock_transformers: ModuleType) -> None:
    from sheaf.backends.bark import BarkBackend

    backend = BarkBackend(model_name="suno/bark", device="cpu")
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()

    mock_transformers.AutoProcessor.from_pretrained.assert_called_once_with("suno/bark")  # type: ignore[attr-defined]
    mock_transformers.BarkModel.from_pretrained.assert_called_once_with("suno/bark")  # type: ignore[attr-defined]


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.bark import BarkBackend

    backend = BarkBackend(model_name="suno/bark-small", device="cuda")
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        backend.load()

    mock_transformers.BarkModel.from_pretrained.return_value.to.assert_called_once_with(
        "cuda"
    )  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_returns_tts_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="bark", text="Hello world.")
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"transformers": mock}):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, TTSResponse)
    assert resp.sample_rate == 24000
    assert resp.audio_b64


def test_predict_audio_b64_is_valid_wav(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="bark", text="Test.")
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"transformers": mock}):
        resp = loaded_backend.predict(req)

    wav_bytes = base64.b64decode(resp.audio_b64)
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"
    assert wav_bytes[12:16] == b"fmt "


def test_predict_voice_preset_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="bark", text="Hello.", voice_preset="v2/en_speaker_6")
    mock = _make_transformers_mod()
    _, processor = _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"transformers": mock}):
        loaded_backend.predict(req)

    processor.assert_called_once_with(
        "Hello.",
        voice_preset="v2/en_speaker_6",
        return_tensors="pt",
    )


def test_predict_none_voice_preset_forwarded(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="bark", text="Hi.")
    mock = _make_transformers_mod()
    _, processor = _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"transformers": mock}):
        loaded_backend.predict(req)

    _, kwargs = processor.call_args
    assert kwargs.get("voice_preset") is None


def test_predict_sample_rate_from_generation_config(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    req = TTSRequest(model_name="bark", text="Hi.")
    mock = _make_transformers_mod(sample_rate=22050)
    _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"transformers": mock}):
        resp = loaded_backend.predict(req)

    assert resp.sample_rate == 22050


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    reqs = [
        TTSRequest(model_name="bark", text="Hello."),
        TTSRequest(model_name="bark", text="World.", voice_preset="v2/en_speaker_3"),
    ]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)

    with patch.dict(sys.modules, {"transformers": mock}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, TTSResponse) for r in responses)
    assert model.generate.call_count == 2
