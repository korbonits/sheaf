"""Unit tests for ImageBindBackend — all imagebind and torch imports are mocked.

No imagebind or torch package required in the test environment.
"""

from __future__ import annotations

import base64
import sys
import types
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.base import ModelType
from sheaf.api.multimodal_embedding import (
    MODALITY_AUDIO,
    MODALITY_DEPTH,
    MODALITY_TEXT,
    MODALITY_THERMAL,
    MODALITY_VISION,
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
)
from sheaf.backends.imagebind import ImageBindBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 1024
N = 2  # number of items per request in most tests


# ---------------------------------------------------------------------------
# Fake ModalityType
# ---------------------------------------------------------------------------


class _MockModalityType:
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    DEPTH = "depth"
    THERMAL = "thermal"


# ---------------------------------------------------------------------------
# Fake tensor — implements the ops used inside _run()
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape  # type: ignore[return-value]

    def norm(self, dim: int, keepdim: bool = False) -> _FakeTensor:
        norms = np.linalg.norm(self._data, axis=dim, keepdims=keepdim)
        return _FakeTensor(norms)

    def __truediv__(self, other: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(self._data / other._data)

    def cpu(self) -> _FakeTensor:
        return self

    def float(self) -> _FakeTensor:
        return _FakeTensor(self._data.astype(np.float32))

    def tolist(self) -> list[list[float]]:
        return self._data.tolist()  # type: ignore[return-value]


def _fake_embeddings(n: int = N, dim: int = DIM) -> _FakeTensor:
    """Return a (n, dim) tensor of unit vectors."""
    data = np.ones((n, dim), dtype=np.float32)
    data /= np.linalg.norm(data, axis=-1, keepdims=True)
    return _FakeTensor(data)


# ---------------------------------------------------------------------------
# Fake torch module — provides no_grad context manager
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def _make_fake_torch() -> ModuleType:
    mod = types.ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fake imagebind package
# ---------------------------------------------------------------------------


def _make_imagebind_mods() -> dict[str, ModuleType]:
    """Build the minimal imagebind sub-module tree."""
    # imagebind.models.imagebind_model
    model_mod = types.ModuleType("imagebind.models.imagebind_model")
    model_mod.ModalityType = _MockModalityType  # type: ignore[attr-defined]

    def _fake_imagebind_huge(pretrained: bool = True) -> MagicMock:
        model = MagicMock()

        def _forward(inputs: dict) -> dict:
            return {k: _fake_embeddings(N, DIM) for k in inputs}

        # side_effect is the correct way to override MagicMock's call behaviour —
        # setting __call__ on the instance doesn't work because Python looks up
        # special methods on the type, not the instance.
        model.side_effect = _forward
        model.eval.return_value = model
        model.to.return_value = model
        return model

    model_mod.imagebind_huge = _fake_imagebind_huge  # type: ignore[attr-defined]

    # imagebind.models
    models_mod = types.ModuleType("imagebind.models")
    models_mod.imagebind_model = model_mod  # type: ignore[attr-defined]

    # imagebind.data
    data_mod = types.ModuleType("imagebind.data")

    def _noop_loader(items: list[str], device: str) -> _FakeTensor:
        return _fake_embeddings(len(items), DIM)

    data_mod.load_and_transform_text = _noop_loader  # type: ignore[attr-defined]
    data_mod.load_and_transform_vision_data = _noop_loader  # type: ignore[attr-defined]
    data_mod.load_and_transform_audio_data = _noop_loader  # type: ignore[attr-defined]
    data_mod.load_and_transform_depth_data = _noop_loader  # type: ignore[attr-defined]
    data_mod.load_and_transform_thermal_data = _noop_loader  # type: ignore[attr-defined]

    # top-level imagebind package
    pkg = types.ModuleType("imagebind")

    return {
        "imagebind": pkg,
        "imagebind.models": models_mod,
        "imagebind.models.imagebind_model": model_mod,
        "imagebind.data": data_mod,
        "torch": _make_fake_torch(),
    }


# Shared sys.modules patch used by tests that call _run()
_FAKE_MODULES = _make_imagebind_mods()


def _loaded_backend() -> ImageBindBackend:
    """Return a loaded ImageBindBackend with all heavy deps mocked."""
    backend = ImageBindBackend(pretrained=False, device="cpu")
    with patch.dict(sys.modules, _FAKE_MODULES):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# API contract tests — no backend required
# ---------------------------------------------------------------------------


def test_request_text_only() -> None:
    req = MultimodalEmbeddingRequest(model_name="ib", texts=["hello", "world"])
    assert req.modality == MODALITY_TEXT
    assert req.n_items == 2


def test_request_images_only() -> None:
    b64 = base64.b64encode(b"fake").decode()
    req = MultimodalEmbeddingRequest(model_name="ib", images_b64=[b64])
    assert req.modality == MODALITY_VISION
    assert req.n_items == 1


def test_request_audio_only() -> None:
    b64 = base64.b64encode(b"fake").decode()
    req = MultimodalEmbeddingRequest(model_name="ib", audios_b64=[b64])
    assert req.modality == MODALITY_AUDIO


def test_request_depth_only() -> None:
    b64 = base64.b64encode(b"fake").decode()
    req = MultimodalEmbeddingRequest(model_name="ib", depth_images_b64=[b64])
    assert req.modality == MODALITY_DEPTH


def test_request_thermal_only() -> None:
    b64 = base64.b64encode(b"fake").decode()
    req = MultimodalEmbeddingRequest(model_name="ib", thermal_images_b64=[b64])
    assert req.modality == MODALITY_THERMAL


def test_request_no_modality_raises() -> None:
    with pytest.raises(Exception, match="Exactly one modality"):
        MultimodalEmbeddingRequest(model_name="ib")


def test_request_two_modalities_raises() -> None:
    b64 = base64.b64encode(b"fake").decode()
    with pytest.raises(Exception, match="Only one modality"):
        MultimodalEmbeddingRequest(model_name="ib", texts=["hello"], images_b64=[b64])


def test_request_invalid_base64_raises() -> None:
    with pytest.raises(Exception):
        MultimodalEmbeddingRequest(model_name="ib", images_b64=["not-valid-base64!!!"])


# ---------------------------------------------------------------------------
# Backend load / import tests
# ---------------------------------------------------------------------------


def test_load_missing_package_raises() -> None:
    """ImportError when imagebind is not installed."""
    import builtins

    _orig = builtins.__import__

    def _block_imagebind(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("imagebind"):
            raise ImportError("imagebind blocked")
        return _orig(name, *args, **kwargs)

    backend = ImageBindBackend(pretrained=False, device="cpu")
    # Remove any cached imagebind mocks so the import hook is reached
    clean = {k: v for k, v in _FAKE_MODULES.items() if not k.startswith("imagebind")}
    with patch.dict(sys.modules, clean, clear=False):
        for key in list(sys.modules):
            if key.startswith("imagebind"):
                del sys.modules[key]
        with patch("builtins.__import__", side_effect=_block_imagebind):
            with pytest.raises(ImportError, match="imagebind"):
                backend.load()


def test_load_sets_model_and_data() -> None:
    backend = ImageBindBackend(pretrained=False, device="cpu")
    with patch.dict(sys.modules, _FAKE_MODULES):
        backend.load()
    assert backend._model is not None
    assert backend._data is not None
    assert backend._ModalityType is _MockModalityType


def test_predict_before_load_raises() -> None:
    backend = ImageBindBackend(pretrained=False, device="cpu")
    req = MultimodalEmbeddingRequest(model_name="ib", texts=["hello"])
    with patch.dict(sys.modules, _FAKE_MODULES):
        with pytest.raises(RuntimeError, match="load\\(\\) first"):
            backend.predict(req)


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------


def test_predict_text() -> None:
    backend = _loaded_backend()
    req = MultimodalEmbeddingRequest(model_name="imagebind", texts=["a dog", "a cat"])
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    assert isinstance(resp, MultimodalEmbeddingResponse)
    assert resp.modality == MODALITY_TEXT
    assert resp.dim == DIM
    assert len(resp.embeddings) == N
    assert len(resp.embeddings[0]) == DIM


def test_predict_vision() -> None:
    backend = _loaded_backend()
    b64 = base64.b64encode(b"\xff\xd8\xff").decode()
    req = MultimodalEmbeddingRequest(model_name="imagebind", images_b64=[b64, b64])
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    assert resp.modality == MODALITY_VISION
    assert resp.dim == DIM
    assert len(resp.embeddings) == N


def test_predict_audio() -> None:
    backend = _loaded_backend()
    b64 = base64.b64encode(b"RIFF" + b"\x00" * 40).decode()
    req = MultimodalEmbeddingRequest(model_name="imagebind", audios_b64=[b64, b64])
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    assert resp.modality == MODALITY_AUDIO
    assert resp.dim == DIM


def test_predict_depth() -> None:
    backend = _loaded_backend()
    b64 = base64.b64encode(b"\x89PNG\r\n").decode()
    req = MultimodalEmbeddingRequest(
        model_name="imagebind", depth_images_b64=[b64, b64]
    )
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    assert resp.modality == MODALITY_DEPTH
    assert resp.dim == DIM


def test_predict_thermal() -> None:
    backend = _loaded_backend()
    b64 = base64.b64encode(b"\xff\xd8\xff").decode()
    req = MultimodalEmbeddingRequest(
        model_name="imagebind", thermal_images_b64=[b64, b64]
    )
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    assert resp.modality == MODALITY_THERMAL
    assert resp.dim == DIM


def test_normalize_true_produces_unit_vectors() -> None:
    backend = _loaded_backend()
    req = MultimodalEmbeddingRequest(
        model_name="imagebind", texts=["hello", "world"], normalize=True
    )
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    for emb in resp.embeddings:
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-4


def test_wrong_request_type_raises() -> None:
    from sheaf.api.time_series import Frequency, TimeSeriesRequest

    backend = _loaded_backend()
    wrong = TimeSeriesRequest(
        model_name="x", history=[1.0, 2.0], horizon=3, frequency=Frequency.HOURLY
    )
    with pytest.raises(TypeError, match="MultimodalEmbeddingRequest"):
        backend.predict(wrong)


def test_response_model_type() -> None:
    backend = _loaded_backend()
    req = MultimodalEmbeddingRequest(model_name="imagebind", texts=["hi"])
    with patch.dict(sys.modules, _FAKE_MODULES):
        resp = backend.predict(req)
    assert resp.model_type == ModelType.MULTIMODAL_EMBEDDING


def test_temp_files_cleaned_up() -> None:
    """Temp files written for image inputs are deleted after inference."""
    import os

    backend = _loaded_backend()
    written: list[str] = []

    from sheaf.backends import imagebind as _ib_mod

    _orig_write = _ib_mod._write_temp_files

    def _tracking_write(items_b64: list[str], suffix: str) -> list[str]:
        paths = _orig_write(items_b64, suffix)
        written.extend(paths)
        return paths

    b64 = base64.b64encode(b"\xff\xd8\xff").decode()
    req = MultimodalEmbeddingRequest(model_name="imagebind", images_b64=[b64])
    with (
        patch.object(_ib_mod, "_write_temp_files", side_effect=_tracking_write),
        patch.dict(sys.modules, _FAKE_MODULES),
    ):
        backend.predict(req)

    for path in written:
        assert not os.path.exists(path), f"Temp file not cleaned up: {path}"
