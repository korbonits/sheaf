"""API contract for weather / atmospheric-state foundation models.

Supports GraphCast, Aurora, Pangu-Weather, and similar architectures.

Encoding convention
-------------------
All array fields are base64-encoded little-endian float32 byte strings.

  Surface variable   shape: (n_lat, n_lon)
  Atmospheric var    shape: (n_levels, n_lat, n_lon)

Encode:  base64.b64encode(arr.astype(np.float32).tobytes()).decode()
Decode:  np.frombuffer(base64.b64decode(s), dtype=np.float32)
         .reshape(n_lat, n_lon)          # surface
         .reshape(n_levels, n_lat, n_lon) # atmospheric

Grid conventions (GraphCast / ERA5)
------------------------------------
- lat:             descending, e.g. [90.0, 89.75, …, -90.0] for 0.25° global
- lon:             ascending,  e.g. [0.0, 0.25, …, 359.75]
- pressure_levels: descending hPa, e.g. [1000, 925, 850, …, 1]
- current_time:    ISO-8601 string, e.g. "2023-01-01T06:00:00"

GraphCast requires two consecutive time steps (t-6h and t) as input, so
both *_vars and prev_*_vars are required and must contain the same variable
names.  n_steps controls how many 6-hour steps are predicted autoregressively.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class WeatherRequest(BaseRequest):
    """Request contract for atmospheric-state foundation models.

    Args:
        surface_vars:        Surface variable fields at time t.
                             Keys are variable names (ERA5-style for GraphCast,
                             e.g. "2m_temperature", "10m_u_component_of_wind").
                             Values are base64 float32 arrays of shape (n_lat, n_lon).
        atmospheric_vars:    Atmospheric (pressure-level) fields at time t.
                             Values are base64 float32 arrays of shape
                             (n_levels, n_lat, n_lon).
        prev_surface_vars:   Same variables at time t - step_hours (t-6h for GraphCast).
        prev_atmospheric_vars: Same variables at time t - step_hours.
        lat:                 Latitude grid, length n_lat, descending degrees.
        lon:                 Longitude grid, length n_lon, ascending degrees.
        pressure_levels:     Pressure levels in hPa, length n_levels, descending.
        current_time:        ISO-8601 timestamp for the *current* state (t).
        n_steps:             Number of autoregressive 6-hour steps to predict.
    """

    model_type: Literal[ModelType.WEATHER] = ModelType.WEATHER

    # Current state (t)
    surface_vars: dict[str, str]
    atmospheric_vars: dict[str, str]

    # Previous state (t - step_hours)
    prev_surface_vars: dict[str, str]
    prev_atmospheric_vars: dict[str, str]

    # Grid metadata
    lat: list[float]
    lon: list[float]
    pressure_levels: list[int]

    # Temporal
    current_time: str  # ISO-8601
    n_steps: int = 1

    @model_validator(mode="after")
    def validate_vars(self) -> WeatherRequest:
        if not self.surface_vars and not self.atmospheric_vars:
            raise ValueError(
                "At least one of surface_vars or atmospheric_vars must be non-empty."
            )
        if set(self.surface_vars) != set(self.prev_surface_vars):
            raise ValueError(
                "surface_vars and prev_surface_vars must contain "
                "the same variable names."
            )
        if set(self.atmospheric_vars) != set(self.prev_atmospheric_vars):
            raise ValueError(
                "atmospheric_vars and prev_atmospheric_vars must contain "
                "the same variable names."
            )
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        return self


class WeatherResponse(BaseResponse):
    """Response contract for atmospheric-state foundation models.

    surface_forecasts[i]    — dict of {var_name: base64_float32} for step i+1.
                              Each array has shape (n_lat, n_lon).
    atmospheric_forecasts[i] — same for atmospheric (pressure-level) variables.
                               Each array has shape (n_levels, n_lat, n_lon).
    forecast_times[i]       — ISO-8601 timestamp for step i+1.
    """

    model_type: Literal[ModelType.WEATHER] = ModelType.WEATHER

    # One entry per predicted step
    surface_forecasts: list[dict[str, str]]
    atmospheric_forecasts: list[dict[str, str]]

    lat: list[float]
    lon: list[float]
    pressure_levels: list[int]
    forecast_times: list[str]  # ISO-8601 per step
    step_hours: int  # 6 for GraphCast
    n_steps: int
