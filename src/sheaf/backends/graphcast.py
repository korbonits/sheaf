"""GraphCast backend for global weather forecasting via Google DeepMind's GraphCast.

Requires: pip install "sheaf-serve[weather]"
Libraries: graphcast (google-deepmind/graphcast), dm-haiku, jax, xarray

Supported checkpoints (download from gs://dm_graphcast/params/):
  "GraphCast_small - ERA5 1959-2015 res=1.0 mesh=2to5 drop=0.1.npz"
      1° global grid, 13 pressure levels — recommended for testing / CPU runs.
  "GraphCast - ERA5 1979-2015 res=0.25 mesh=2to6 drop=0.1.npz"
      0.25° global grid, 37 pressure levels — full operational resolution.
  "GraphCast_operational - ERA5-HRES 1979-2021 res=0.25 mesh=2to6 drop=0.1.npz"
      0.25° grid, operational ERA5-HRES inputs (fewer pressure levels).

Inputs (via WeatherRequest)
---------------------------
Two consecutive 6-hour ERA5 states are required:
  surface_vars / prev_surface_vars      — ERA5 surface variables
  atmospheric_vars / prev_atmospheric_vars — ERA5 pressure-level variables

Standard ERA5 surface variable names (examples):
  "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
  "mean_sea_level_pressure", "total_precipitation_6hr",
  "10m_wind_speed", "geopotential_at_surface", "land_sea_mask"

Standard ERA5 atmospheric variable names (examples):
  "temperature", "u_component_of_wind", "v_component_of_wind",
  "geopotential", "specific_humidity", "vertical_velocity"

Checkpoint path
---------------
Pass checkpoint_path=".../GraphCast_small.npz" (or the full path) when
constructing the backend. GraphCast checkpoint files are NumPy archive files
containing {"params": ..., "model_config": ..., "task_config": ...}.

Autoregressive rollout
----------------------
n_steps=1 predicts t+6h; n_steps=4 predicts t+6h, t+12h, t+18h, t+24h.
Each additional step adds one full forward pass.

Solar-radiation forcings are computed internally from the target timestamps
using graphcast's solar_radiation module when available; they are omitted
(empty Dataset) if the module is unavailable, which is acceptable for small
checkpoints that don't use them.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.weather import WeatherRequest, WeatherResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_STEP_HOURS = 6  # GraphCast operates on 6-hour increments


@register_backend("graphcast")
class GraphCastBackend(ModelBackend):
    """ModelBackend for GraphCast global weather forecasting.

    Loads a GraphCast checkpoint (NumPy archive), JIT-compiles the forward
    pass with JAX/Haiku at load() time, then serves autoregressive predictions
    through predict() / batch_predict().

    Args:
        checkpoint_path: Path to a GraphCast .npz checkpoint file.
        device:          JAX device string: "cpu", "gpu", or "tpu".
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._params: Any = None
        self._state: dict = {}
        self._model_config: Any = None
        self._task_config: Any = None
        self._run_fn: Any = None  # JIT-compiled forward pass
        self._rollout: Any = None  # graphcast.rollout module
        self._xr: Any = None  # xarray module, stored at load() for testability
        self._jax: Any = None  # jax module

    @property
    def model_type(self) -> str:
        return ModelType.WEATHER

    def load(self) -> None:
        try:
            import haiku as hk  # ty: ignore[unresolved-import]
            import jax  # ty: ignore[unresolved-import]
            import xarray as xr  # ty: ignore[unresolved-import]
            from graphcast import (  # ty: ignore[unresolved-import]
                checkpoint as gc_checkpoint,
            )
            from graphcast import graphcast as gc  # ty: ignore[unresolved-import]
            from graphcast import rollout  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "graphcast, dm-haiku, jax, and xarray are required for the "
                "GraphCast backend. Install with: "
                "pip install 'sheaf-serve[weather]'"
            ) from e

        self._xr = xr
        self._jax = jax

        # Load checkpoint: params + model/task configs
        with open(self._checkpoint_path, "rb") as f:
            ckpt = gc_checkpoint.load(f, gc.CheckPoint)

        self._model_config = ckpt.model_config
        self._task_config = ckpt.task_config
        self._params = ckpt.params
        self._state = {}
        self._rollout = rollout

        # Build the Haiku-transformed forward function and JIT-compile.
        # This traces the graph once at load() time so predict() has no
        # compilation overhead.
        @hk.transform_with_state
        def _forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = gc.GraphCast(model_config, task_config)
            return predictor(inputs, targets_template, forcings, is_training=False)

        self._run_fn = jax.jit(_forward.apply)

    # ------------------------------------------------------------------
    # Public inference interface
    # ------------------------------------------------------------------

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, WeatherRequest):
            raise TypeError(f"Expected WeatherRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _run(self, request: WeatherRequest) -> WeatherResponse:
        if self._run_fn is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        inputs = self._build_inputs(request)
        targets_template = self._build_targets_template(inputs, request.n_steps)
        forcings = self._build_forcings(inputs, targets_template)

        predictions = self._rollout.chunked_prediction(
            self._run_fn,
            rng=self._jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )

        return self._build_response(predictions, request)

    # ------------------------------------------------------------------
    # Array encode / decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_array(b64: str, shape: tuple[int, ...]) -> Any:
        import numpy as np

        return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(shape)

    @staticmethod
    def _encode_array(arr: Any) -> str:
        import numpy as np

        return base64.b64encode(np.asarray(arr, dtype=np.float32).tobytes()).decode()

    # ------------------------------------------------------------------
    # xarray Dataset construction
    # ------------------------------------------------------------------

    def _build_inputs(self, request: WeatherRequest) -> Any:
        """Decode request arrays → xarray.Dataset for GraphCast inputs.

        GraphCast expects an inputs Dataset with:
          - surface vars:     dims (batch=1, time=2, lat, lon)
          - atmospheric vars: dims (batch=1, time=2, level, lat, lon)
          - time coordinate:  [t-6h, t] as datetime64
        """
        import numpy as np

        xr = self._xr
        n_lat = len(request.lat)
        n_lon = len(request.lon)
        n_lev = len(request.pressure_levels)
        lat = np.array(request.lat, dtype=np.float32)
        lon = np.array(request.lon, dtype=np.float32)
        levels = np.array(request.pressure_levels, dtype=np.int32)

        current_dt = datetime.fromisoformat(request.current_time)
        prev_dt = current_dt - timedelta(hours=_STEP_HOURS)
        times = [
            np.datetime64(prev_dt.isoformat()),
            np.datetime64(current_dt.isoformat()),
        ]

        data_vars: dict[str, Any] = {}

        for var_name in request.surface_vars:
            prev_arr = self._decode_array(
                request.prev_surface_vars[var_name], (n_lat, n_lon)
            )
            curr_arr = self._decode_array(
                request.surface_vars[var_name], (n_lat, n_lon)
            )
            # (batch=1, time=2, lat, lon)
            stacked = np.stack([prev_arr, curr_arr], axis=0)[np.newaxis]
            data_vars[var_name] = xr.DataArray(
                stacked, dims=["batch", "time", "lat", "lon"]
            )

        for var_name in request.atmospheric_vars:
            prev_arr = self._decode_array(
                request.prev_atmospheric_vars[var_name], (n_lev, n_lat, n_lon)
            )
            curr_arr = self._decode_array(
                request.atmospheric_vars[var_name], (n_lev, n_lat, n_lon)
            )
            # (batch=1, time=2, level, lat, lon)
            stacked = np.stack([prev_arr, curr_arr], axis=0)[np.newaxis]
            data_vars[var_name] = xr.DataArray(
                stacked, dims=["batch", "time", "level", "lat", "lon"]
            )

        return xr.Dataset(
            data_vars,
            coords={"lat": lat, "lon": lon, "level": levels, "time": times},
        )

    def _build_targets_template(self, inputs: Any, n_steps: int) -> Any:
        """Build an empty target Dataset with the right shape for n_steps ahead."""
        import numpy as np

        xr = self._xr
        last_time = inputs.coords["time"].values[-1]
        target_times = [
            last_time + np.timedelta64(_STEP_HOURS * i, "h")
            for i in range(1, n_steps + 1)
        ]

        n_lat = len(inputs.coords["lat"])
        n_lon = len(inputs.coords["lon"])
        n_lev = len(inputs.coords["level"])

        data_vars: dict[str, Any] = {}
        for var_name, da in inputs.data_vars.items():
            if "level" in da.dims:
                data_vars[var_name] = xr.DataArray(
                    np.zeros((1, n_steps, n_lev, n_lat, n_lon), dtype=np.float32),
                    dims=["batch", "time", "level", "lat", "lon"],
                )
            else:
                data_vars[var_name] = xr.DataArray(
                    np.zeros((1, n_steps, n_lat, n_lon), dtype=np.float32),
                    dims=["batch", "time", "lat", "lon"],
                )

        return xr.Dataset(
            data_vars,
            coords={
                "lat": inputs.coords["lat"],
                "lon": inputs.coords["lon"],
                "level": inputs.coords["level"],
                "time": target_times,
            },
        )

    def _build_forcings(self, inputs: Any, targets_template: Any) -> Any:
        """Build forcing Dataset (solar radiation at target times).

        Attempts to use graphcast.solar_radiation if available; falls back to
        an empty Dataset (acceptable for checkpoints that don't use forcings).
        """
        xr = self._xr
        try:
            from graphcast import (  # ty: ignore[unresolved-import]
                solar_radiation,
            )

            return solar_radiation.get_toa_incident_solar_radiation_for_xarray(
                targets_template
            )
        except Exception:
            return xr.Dataset(
                coords={
                    "lat": inputs.coords["lat"],
                    "lon": inputs.coords["lon"],
                    "time": targets_template.coords["time"],
                }
            )

    def _build_response(
        self, predictions: Any, request: WeatherRequest
    ) -> WeatherResponse:
        """Convert xarray predictions Dataset → WeatherResponse."""
        import numpy as np

        current_dt = datetime.fromisoformat(request.current_time)

        surface_forecasts: list[dict[str, str]] = []
        atmospheric_forecasts: list[dict[str, str]] = []
        forecast_times: list[str] = []

        for i in range(request.n_steps):
            step_dt = current_dt + timedelta(hours=_STEP_HOURS * (i + 1))
            forecast_times.append(step_dt.isoformat())

            surf: dict[str, str] = {}
            atmos: dict[str, str] = {}

            for var_name in predictions.data_vars:
                # shape: (batch=1, time=n_steps, [level,] lat, lon)
                da = predictions[var_name]
                arr = np.asarray(da.values[0, i])  # remove batch dim, select step i
                if "level" in da.dims:
                    atmos[var_name] = self._encode_array(arr)
                else:
                    surf[var_name] = self._encode_array(arr)

            surface_forecasts.append(surf)
            atmospheric_forecasts.append(atmos)

        return WeatherResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            surface_forecasts=surface_forecasts,
            atmospheric_forecasts=atmospheric_forecasts,
            lat=request.lat,
            lon=request.lon,
            pressure_levels=request.pressure_levels,
            forecast_times=forecast_times,
            step_hours=_STEP_HOURS,
            n_steps=request.n_steps,
        )
