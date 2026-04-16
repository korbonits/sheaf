"""Tests for PrithviBackend — fully mocked, no transformers or torch required.

Covers:
  - load() raises ImportError when transformers is absent
  - load() passes model_name and trust_remote_code to from_pretrained
  - load() moves model to specified device
  - predict() rejects non-SatelliteRequest inputs
  - predict() returns SatelliteResponse with correct structure
  - predict() dim matches hidden_size of fake model output
  - predict() mean pooling averages all tokens
  - predict() cls pooling takes the first token
  - predict() n_time passed through to response
  - predict() applies per-band normalization when normalize=True
  - predict() skips normalization when normalize=False
  - predict() skips normalization when band count mismatches processor stats
  - _normalize handles missing processor stats gracefully
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

from sheaf.api.satellite import SatelliteRequest, SatelliteResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_TIME = 2
_N_BANDS = 6
_H, _W = 16, 16
_DIM = 8  # fake hidden size
_BAND_NAMES = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def _enc(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _zero_pixels(n_time: int = _N_TIME) -> str:
    return _enc(np.zeros((n_time, _N_BANDS, _H, _W), dtype=np.float32))


def _const_pixels(v: float, n_time: int = _N_TIME) -> str:
    return _enc(np.full((n_time, _N_BANDS, _H, _W), v, dtype=np.float32))


# ---------------------------------------------------------------------------
# FakeTensor
# ---------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, data: list | np.ndarray) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        return FakeTensor(self._data[key])

    def mean(self, dim: int) -> FakeTensor:
        return FakeTensor(self._data.mean(axis=dim))

    def cpu(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def tolist(self) -> list:
        return self._data.tolist()

    def unsqueeze(self, dim: int) -> FakeTensor:
        return FakeTensor(np.expand_dims(self._data, axis=dim))

    def to(self, device: str) -> FakeTensor:  # noqa: ARG002
        return self


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _make_torch_mod() -> ModuleType:
    mod = ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]

    def _from_numpy(arr: np.ndarray) -> FakeTensor:
        return FakeTensor(arr)

    mod.from_numpy = _from_numpy  # type: ignore[attr-defined]
    return mod


_torch_mod = _make_torch_mod()

# ---------------------------------------------------------------------------
# Fake transformers module
# ---------------------------------------------------------------------------


def _make_model_output(n_tokens: int = 32, hidden_size: int = _DIM) -> MagicMock:
    hidden = FakeTensor(np.ones((1, n_tokens, hidden_size), dtype=np.float32))
    output = MagicMock()
    output.last_hidden_state = hidden
    return output


def _make_transformers_mod(
    hidden_size: int = _DIM,
    n_tokens: int = 32,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
) -> ModuleType:
    if image_mean is None:
        image_mean = [0.5] * _N_BANDS
    if image_std is None:
        image_std = [0.1] * _N_BANDS

    model_output = _make_model_output(n_tokens=n_tokens, hidden_size=hidden_size)
    model = MagicMock()
    model.return_value = model_output
    model.to.return_value = model
    model.eval.return_value = None

    processor = MagicMock()
    processor.image_mean = image_mean
    processor.image_std = image_std

    mod = ModuleType("transformers")
    mod.AutoModel = MagicMock()  # type: ignore[attr-defined]
    mod.AutoModel.from_pretrained.return_value = model  # type: ignore[attr-defined]
    mod.AutoImageProcessor = MagicMock()  # type: ignore[attr-defined]
    mod.AutoImageProcessor.from_pretrained.return_value = processor  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transformers() -> ModuleType:
    return _make_transformers_mod()


@pytest.fixture
def loaded_backend(mock_transformers: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.prithvi import PrithviBackend

    backend = PrithviBackend(
        model_name="ibm-nasa-geospatial/Prithvi-EO-2.0-300M", device="cpu"
    )
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()
    return backend


def _wire(backend, mod: ModuleType) -> tuple[MagicMock, MagicMock]:
    model = mod.AutoModel.from_pretrained.return_value
    processor = mod.AutoImageProcessor.from_pretrained.return_value
    backend._model = model
    backend._processor = processor
    return model, processor


# ---------------------------------------------------------------------------
# Request factory
# ---------------------------------------------------------------------------


def _make_request(
    n_time: int = _N_TIME,
    pooling: str = "mean",
    normalize: bool = False,
) -> SatelliteRequest:
    return SatelliteRequest(
        model_name="prithvi",
        pixels_b64=_zero_pixels(n_time),
        n_time=n_time,
        n_bands=_N_BANDS,
        height=_H,
        width=_W,
        band_names=_BAND_NAMES,
        pooling=pooling,  # type: ignore[arg-type]
        normalize=normalize,
    )


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_transformers() -> None:
    from sheaf.backends.prithvi import PrithviBackend

    backend = PrithviBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "transformers" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name == "transformers":
            raise ModuleNotFoundError("No module named 'transformers'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[earth-observation\\]"),
    ):
        backend.load()


def test_load_passes_model_name_and_trust_remote_code(
    mock_transformers: ModuleType,
) -> None:
    from sheaf.backends.prithvi import PrithviBackend

    backend = PrithviBackend(model_name="ibm-nasa-geospatial/Prithvi-EO-1.0-100M")
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    mock_transformers.AutoModel.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "ibm-nasa-geospatial/Prithvi-EO-1.0-100M", trust_remote_code=True
    )
    mock_transformers.AutoImageProcessor.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "ibm-nasa-geospatial/Prithvi-EO-1.0-100M", trust_remote_code=True
    )


def test_load_moves_model_to_device(mock_transformers: ModuleType) -> None:
    from sheaf.backends.prithvi import PrithviBackend

    backend = PrithviBackend(device="cuda")
    _mods = {"transformers": mock_transformers, "torch": _torch_mod}
    with patch.dict(sys.modules, _mods):
        backend.load()

    mock_transformers.AutoModel.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="SatelliteRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_satellite_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert isinstance(resp, SatelliteResponse)
    assert resp.dim == _DIM
    assert len(resp.embedding) == _DIM
    assert resp.n_time == _N_TIME


def test_predict_dim_matches_hidden_size(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    hidden_size = 16
    mock = _make_transformers_mod(hidden_size=hidden_size)
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request())

    assert resp.dim == hidden_size
    assert len(resp.embedding) == hidden_size


def test_predict_n_time_passthrough(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(n_time=3))

    assert resp.n_time == 3


# ---------------------------------------------------------------------------
# predict() — pooling
# ---------------------------------------------------------------------------


def test_predict_mean_pooling_averages_all_tokens(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """All tokens are equal (all-ones), so mean == first token == any token."""
    n_tokens, hidden_size = 10, 4
    mock = _make_transformers_mod(n_tokens=n_tokens, hidden_size=hidden_size)
    _wire(loaded_backend, mock)
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(pooling="mean"))

    # All tokens are 1.0, so mean = 1.0 in every dim
    assert resp.embedding == pytest.approx([1.0] * hidden_size)


def test_predict_cls_pooling_takes_first_token(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """CLS pooling should take index-0 token, which is distinct from later ones."""
    n_tokens, hidden_size = 5, 4
    # Build a model whose first token is all-2s, rest are all-0s
    hidden_data = np.zeros((1, n_tokens, hidden_size), dtype=np.float32)
    hidden_data[0, 0, :] = 2.0
    model_output = MagicMock()
    model_output.last_hidden_state = FakeTensor(hidden_data)
    mock = _make_transformers_mod(n_tokens=n_tokens, hidden_size=hidden_size)
    mock.AutoModel.from_pretrained.return_value.return_value = model_output
    loaded_backend._model = mock.AutoModel.from_pretrained.return_value

    with patch.dict(sys.modules, {"torch": _torch_mod}):
        resp = loaded_backend.predict(_make_request(pooling="cls"))

    assert resp.embedding == pytest.approx([2.0] * hidden_size)


# ---------------------------------------------------------------------------
# predict() — normalization
# ---------------------------------------------------------------------------


def test_predict_normalize_true_applies_zscore(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """When normalize=True the pixels should be z-scored per band."""
    means = [0.5] * _N_BANDS
    stds = [0.1] * _N_BANDS
    mock = _make_transformers_mod(image_mean=means, image_std=stds)
    model, processor = _wire(loaded_backend, mock)

    # Capture the pixel_values passed to the model
    captured: list[object] = []

    def _capture(**kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs["pixel_values"])
        return model.return_value

    model.side_effect = _capture

    pixels_val = 0.7  # constant across all bands
    req = SatelliteRequest(
        model_name="prithvi",
        pixels_b64=_const_pixels(pixels_val),
        n_time=_N_TIME,
        n_bands=_N_BANDS,
        height=_H,
        width=_W,
        band_names=_BAND_NAMES,
        normalize=True,
    )
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(req)

    assert len(captured) == 1
    passed = captured[0]
    # Expected normalized value: (0.7 - 0.5) / 0.1 = 2.0
    arr = np.asarray(passed._data, dtype=np.float32)  # type: ignore[union-attr]
    assert arr.mean() == pytest.approx(2.0, abs=1e-4)


def test_predict_normalize_false_skips_zscore(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """When normalize=False raw pixels are passed to the model unchanged."""
    mock = _make_transformers_mod()
    model, processor = _wire(loaded_backend, mock)

    captured: list[object] = []

    def _capture(**kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs["pixel_values"])
        return model.return_value

    model.side_effect = _capture

    pixels_val = 0.7
    req = _make_request(normalize=False)
    req = SatelliteRequest(
        model_name="prithvi",
        pixels_b64=_const_pixels(pixels_val),
        n_time=_N_TIME,
        n_bands=_N_BANDS,
        height=_H,
        width=_W,
        band_names=_BAND_NAMES,
        normalize=False,
    )
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        loaded_backend.predict(req)

    arr = np.asarray(captured[0]._data, dtype=np.float32)  # type: ignore[union-attr]
    assert arr.mean() == pytest.approx(pixels_val, abs=1e-4)


def test_normalize_skipped_on_band_count_mismatch(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """If processor stats have different band count, pixels pass through unchanged."""
    # Processor has stats for 3 bands, request has 6 bands
    mock = _make_transformers_mod(image_mean=[0.5, 0.5, 0.5], image_std=[0.1, 0.1, 0.1])
    _, processor = _wire(loaded_backend, mock)

    pixels = np.full((_N_TIME, _N_BANDS, _H, _W), 0.7, dtype=np.float32)
    result = loaded_backend._normalize(pixels)

    # Should be unchanged since band counts don't match
    np.testing.assert_array_almost_equal(result, pixels)


def test_normalize_graceful_on_missing_processor_stats() -> None:
    """_normalize returns pixels unchanged if processor lacks image_mean/std."""
    from sheaf.backends.prithvi import PrithviBackend

    backend = PrithviBackend()
    backend._processor = MagicMock(spec=[])  # no image_mean / image_std attrs
    pixels = np.ones((_N_TIME, _N_BANDS, _H, _W), dtype=np.float32)
    result = backend._normalize(pixels)
    np.testing.assert_array_equal(result, pixels)


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_independently(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    mock = _make_transformers_mod()
    model, _ = _wire(loaded_backend, mock)

    reqs = [_make_request(), _make_request()]
    with patch.dict(sys.modules, {"torch": _torch_mod}):
        responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, SatelliteResponse) for r in responses)
    assert model.call_count == 2
