"""Tests for GraphCastBackend — fully mocked, no graphcast/JAX/xarray required.

Covers:
  - load() raises ImportError when graphcast is absent
  - load() reads checkpoint and populates model_config / task_config
  - load() JIT-compiles the forward function
  - predict() rejects non-WeatherRequest inputs
  - predict() returns WeatherResponse with correct structure
  - predict() decodes surface and atmospheric arrays correctly
  - predict() encodes output arrays correctly
  - predict() computes forecast_times correctly for n_steps > 1
  - predict() passes lat / lon / pressure_levels through to response
  - _decode_array / _encode_array round-trip
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import builtins
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from sheaf.api.weather import WeatherRequest, WeatherResponse

# ---------------------------------------------------------------------------
# Small synthetic grid
# ---------------------------------------------------------------------------

_N_LAT = 4
_N_LON = 8
_N_LEV = 3
_LAT = [90.0, 60.0, 30.0, 0.0]
_LON = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
_LEVELS = [1000, 500, 100]
_CURRENT_TIME = "2023-01-01T12:00:00"

# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def _enc(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _zero_surface() -> str:
    return _enc(np.zeros((_N_LAT, _N_LON), dtype=np.float32))


def _zero_atmos() -> str:
    return _enc(np.zeros((_N_LEV, _N_LAT, _N_LON), dtype=np.float32))


def _const_surface(v: float) -> str:
    return _enc(np.full((_N_LAT, _N_LON), v, dtype=np.float32))


def _const_atmos(v: float) -> str:
    return _enc(np.full((_N_LEV, _N_LAT, _N_LON), v, dtype=np.float32))


# ---------------------------------------------------------------------------
# Minimal fake xarray
# ---------------------------------------------------------------------------


class FakeDataArray:
    """Minimal xarray.DataArray stand-in."""

    def __init__(
        self,
        data: np.ndarray,
        dims: list[str] | None = None,
    ) -> None:
        self.values = np.asarray(data, dtype=np.float32)
        self.dims = dims or []

    def __getitem__(self, key: Any) -> FakeDataArray:
        return FakeDataArray(self.values[key], dims=self.dims[len(np.shape(key)) :])


class FakeCoords(dict):
    """dict subclass whose values expose a .values attribute."""

    class _CoordVal:
        def __init__(self, v: Any) -> None:
            self.values = np.asarray(v)

        def __len__(self) -> int:
            return len(self.values)

    def __getitem__(self, key: str) -> _CoordVal:
        return FakeCoords._CoordVal(super().__getitem__(key))


class FakeDataset:
    """Minimal xarray.Dataset stand-in."""

    def __init__(
        self,
        data_vars: dict[str, FakeDataArray] | None = None,
        coords: dict[str, Any] | None = None,
    ) -> None:
        self.data_vars = data_vars or {}
        self.coords = FakeCoords(coords or {})

    def __getitem__(self, key: str) -> FakeDataArray:
        return self.data_vars[key]


def _make_xarray_mod() -> ModuleType:
    """Build a fake xarray module backed by FakeDataset / FakeDataArray."""
    mod = ModuleType("xarray")
    mod.DataArray = FakeDataArray  # type: ignore[attr-defined]
    mod.Dataset = FakeDataset  # type: ignore[attr-defined]
    return mod


_xr_mod = _make_xarray_mod()

# ---------------------------------------------------------------------------
# Fake JAX
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _make_jax_mod() -> ModuleType:
    jax_mod = ModuleType("jax")
    jax_mod.jit = lambda fn: fn  # type: ignore[attr-defined]
    jax_mod.random = MagicMock()  # type: ignore[attr-defined]
    jax_mod.random.PRNGKey.return_value = 0
    return jax_mod


_jax_mod = _make_jax_mod()

# ---------------------------------------------------------------------------
# Fake graphcast module tree
# ---------------------------------------------------------------------------


def _make_graphcast_mods(
    n_lat: int = _N_LAT,
    n_lon: int = _N_LON,
    n_lev: int = _N_LEV,
    n_steps: int = 1,
) -> dict[str, ModuleType]:
    """Build fake graphcast.*, haiku, xarray, and jax sys.modules entries."""

    # --- fake CheckPoint ---
    class FakeCheckPoint:
        params: dict = {}
        model_config: Any = MagicMock()
        task_config: Any = MagicMock()
        description: str = "test"
        license: str = "test"

    # --- fake checkpoint module ---
    gc_checkpoint = ModuleType("graphcast.checkpoint")
    gc_checkpoint.load = MagicMock(return_value=FakeCheckPoint())  # type: ignore[attr-defined]

    # --- fake GraphCast class ---
    gc_graphcast_mod = ModuleType("graphcast.graphcast")
    gc_graphcast_mod.CheckPoint = FakeCheckPoint  # type: ignore[attr-defined]
    gc_graphcast_mod.GraphCast = MagicMock()  # type: ignore[attr-defined]

    # --- fake rollout that returns FakeDataset with zeros ---
    def _fake_chunked_prediction(fn, rng, inputs, targets_template, forcings):
        surface_da = FakeDataArray(
            np.zeros((1, n_steps, n_lat, n_lon), dtype=np.float32),
            dims=["batch", "time", "lat", "lon"],
        )
        atmos_da = FakeDataArray(
            np.zeros((1, n_steps, n_lev, n_lat, n_lon), dtype=np.float32),
            dims=["batch", "time", "level", "lat", "lon"],
        )
        return FakeDataset(
            data_vars={"2m_temperature": surface_da, "temperature": atmos_da},
            coords={
                "lat": _LAT,
                "lon": _LON,
                "level": _LEVELS,
                "time": np.array(
                    [
                        np.datetime64("2023-01-01T18:00:00")
                        + np.timedelta64(6 * i, "h")
                        for i in range(n_steps)
                    ]
                ),
            },
        )

    gc_rollout = ModuleType("graphcast.rollout")
    gc_rollout.chunked_prediction = _fake_chunked_prediction  # type: ignore[attr-defined]

    # --- top-level graphcast ---
    gc_top = ModuleType("graphcast")

    # --- fake haiku ---
    hk_mod = ModuleType("haiku")

    class _FakeTransform:
        def apply(self, *a, **kw):  # type: ignore[no-untyped-def]
            return MagicMock(), {}

    def _transform_with_state(fn):  # type: ignore[no-untyped-def]
        return _FakeTransform()

    hk_mod.transform_with_state = _transform_with_state  # type: ignore[attr-defined]

    return {
        "graphcast": gc_top,
        "graphcast.checkpoint": gc_checkpoint,
        "graphcast.graphcast": gc_graphcast_mod,
        "graphcast.rollout": gc_rollout,
        "haiku": hk_mod,
        "xarray": _xr_mod,
        "jax": _jax_mod,
    }


# ---------------------------------------------------------------------------
# Request factory
# ---------------------------------------------------------------------------


def _make_request(n_steps: int = 1) -> WeatherRequest:
    return WeatherRequest(
        model_name="graphcast",
        surface_vars={"2m_temperature": _zero_surface()},
        atmospheric_vars={"temperature": _zero_atmos()},
        prev_surface_vars={"2m_temperature": _zero_surface()},
        prev_atmospheric_vars={"temperature": _zero_atmos()},
        lat=_LAT,
        lon=_LON,
        pressure_levels=_LEVELS,
        current_time=_CURRENT_TIME,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gc_mods() -> dict[str, ModuleType]:
    return _make_graphcast_mods()


@pytest.fixture
def loaded_backend(gc_mods: dict[str, ModuleType]):  # type: ignore[no-untyped-def]
    from sheaf.backends.graphcast import GraphCastBackend

    backend = GraphCastBackend(checkpoint_path="/fake/checkpoint.npz")
    with patch.dict(sys.modules, gc_mods), patch("builtins.open", mock_open()):
        backend.load()
    # Inject fake xarray and jax so _run methods can use them without patching
    backend._xr = _xr_mod
    backend._jax = _jax_mod
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_graphcast() -> None:
    from sheaf.backends.graphcast import GraphCastBackend

    backend = GraphCastBackend(checkpoint_path="/fake/checkpoint.npz")
    mods_without = {
        k: v
        for k, v in sys.modules.items()
        if not any(k.startswith(p) for p in ("graphcast", "haiku", "xarray"))
    }
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name in ("graphcast", "haiku", "xarray"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[weather\\]"),
    ):
        backend.load()


def test_load_reads_checkpoint(gc_mods: dict[str, ModuleType]) -> None:
    from sheaf.backends.graphcast import GraphCastBackend

    backend = GraphCastBackend(checkpoint_path="/fake/checkpoint.npz")
    with patch.dict(sys.modules, gc_mods), patch("builtins.open", mock_open()):
        backend.load()

    assert backend._params is not None
    assert backend._model_config is not None
    assert backend._task_config is not None


def test_load_compiles_run_fn(gc_mods: dict[str, ModuleType]) -> None:
    from sheaf.backends.graphcast import GraphCastBackend

    backend = GraphCastBackend(checkpoint_path="/fake/checkpoint.npz")
    with patch.dict(sys.modules, gc_mods), patch("builtins.open", mock_open()):
        backend.load()

    assert backend._run_fn is not None


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with pytest.raises(TypeError, match="WeatherRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_weather_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=1)
    resp = loaded_backend.predict(req)

    assert isinstance(resp, WeatherResponse)
    assert resp.n_steps == 1
    assert resp.step_hours == 6
    assert len(resp.surface_forecasts) == 1
    assert len(resp.atmospheric_forecasts) == 1
    assert len(resp.forecast_times) == 1


def test_predict_forecast_times_single_step(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=1)
    resp = loaded_backend.predict(req)

    # 2023-01-01T12:00:00 + 6h = 2023-01-01T18:00:00
    assert resp.forecast_times[0] == "2023-01-01T18:00:00"


def test_predict_forecast_times_multi_step(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    gc_mods = _make_graphcast_mods(n_steps=4)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=4)
    resp = loaded_backend.predict(req)

    assert resp.n_steps == 4
    assert len(resp.forecast_times) == 4
    assert resp.forecast_times[0] == "2023-01-01T18:00:00"
    assert resp.forecast_times[1] == "2023-01-02T00:00:00"
    assert resp.forecast_times[2] == "2023-01-02T06:00:00"
    assert resp.forecast_times[3] == "2023-01-02T12:00:00"


def test_predict_passes_grid_through(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=1)
    resp = loaded_backend.predict(req)

    assert resp.lat == _LAT
    assert resp.lon == _LON
    assert resp.pressure_levels == _LEVELS


def test_predict_surface_var_names(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=1)
    resp = loaded_backend.predict(req)

    assert "2m_temperature" in resp.surface_forecasts[0]


def test_predict_atmospheric_var_names(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=1)
    resp = loaded_backend.predict(req)

    assert "temperature" in resp.atmospheric_forecasts[0]


def test_predict_output_arrays_decodable(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """Encoded output arrays must decode to correct shapes."""
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    req = _make_request(n_steps=1)
    resp = loaded_backend.predict(req)

    surf_b64 = resp.surface_forecasts[0]["2m_temperature"]
    surf_arr = np.frombuffer(base64.b64decode(surf_b64), dtype=np.float32).reshape(
        _N_LAT, _N_LON
    )
    assert surf_arr.shape == (_N_LAT, _N_LON)

    atmos_b64 = resp.atmospheric_forecasts[0]["temperature"]
    atmos_arr = np.frombuffer(base64.b64decode(atmos_b64), dtype=np.float32).reshape(
        _N_LEV, _N_LAT, _N_LON
    )
    assert atmos_arr.shape == (_N_LEV, _N_LAT, _N_LON)


# ---------------------------------------------------------------------------
# _decode_array / _encode_array round-trip
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip_surface() -> None:
    from sheaf.backends.graphcast import GraphCastBackend

    arr = np.random.rand(_N_LAT, _N_LON).astype(np.float32)
    b64 = GraphCastBackend._encode_array(arr)
    recovered = GraphCastBackend._decode_array(b64, (_N_LAT, _N_LON))
    np.testing.assert_array_almost_equal(arr, recovered)


def test_encode_decode_roundtrip_atmospheric() -> None:
    from sheaf.backends.graphcast import GraphCastBackend

    arr = np.random.rand(_N_LEV, _N_LAT, _N_LON).astype(np.float32)
    b64 = GraphCastBackend._encode_array(arr)
    recovered = GraphCastBackend._decode_array(b64, (_N_LEV, _N_LAT, _N_LON))
    np.testing.assert_array_almost_equal(arr, recovered)


# ---------------------------------------------------------------------------
# _build_inputs — verifies stacking of prev + current arrays
# ---------------------------------------------------------------------------


def test_build_inputs_surface_values_stacked(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """prev and current surface arrays are stacked as time dim 0 and 1."""
    prev_val, curr_val = 1.0, 2.0
    req = WeatherRequest(
        model_name="graphcast",
        surface_vars={"2m_temperature": _const_surface(curr_val)},
        atmospheric_vars={"temperature": _zero_atmos()},
        prev_surface_vars={"2m_temperature": _const_surface(prev_val)},
        prev_atmospheric_vars={"temperature": _zero_atmos()},
        lat=_LAT,
        lon=_LON,
        pressure_levels=_LEVELS,
        current_time=_CURRENT_TIME,
    )
    inputs = loaded_backend._build_inputs(req)
    da = inputs["2m_temperature"]
    # da.values shape: (batch=1, time=2, lat, lon)
    assert da.values[0, 0].mean() == pytest.approx(prev_val)
    assert da.values[0, 1].mean() == pytest.approx(curr_val)


def test_build_inputs_atmospheric_values_stacked(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    prev_val, curr_val = 3.0, 4.0
    req = WeatherRequest(
        model_name="graphcast",
        surface_vars={"2m_temperature": _zero_surface()},
        atmospheric_vars={"temperature": _const_atmos(curr_val)},
        prev_surface_vars={"2m_temperature": _zero_surface()},
        prev_atmospheric_vars={"temperature": _const_atmos(prev_val)},
        lat=_LAT,
        lon=_LON,
        pressure_levels=_LEVELS,
        current_time=_CURRENT_TIME,
    )
    inputs = loaded_backend._build_inputs(req)
    da = inputs["temperature"]
    assert da.values[0, 0].mean() == pytest.approx(prev_val)
    assert da.values[0, 1].mean() == pytest.approx(curr_val)


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_returns_one_response_per_request(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    gc_mods = _make_graphcast_mods(n_steps=1)
    loaded_backend._rollout = gc_mods["graphcast.rollout"]
    reqs = [_make_request(), _make_request()]
    responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, WeatherResponse) for r in responses)
