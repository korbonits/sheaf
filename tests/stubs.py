"""Stub backends for testing — no test-framework dependency.

Kept in a separate module so cloudpickle can import this file in Ray
worker processes (which don't have pytest installed).
"""

from __future__ import annotations

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("_smoke_ts")
class SmokeTimeSeriesBackend(ModelBackend):
    """Echo backend: returns 0.42 for every horizon step."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, TimeSeriesRequest)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[0.42] * request.horizon,
        )


@register_backend("_smoke_ts_registry")
class SmokeTimeSeriesRegistryBackend(ModelBackend):
    """Like SmokeTimeSeriesBackend but deployed via string registry lookup.

    Used to verify that SHEAF_EXTRA_BACKENDS causes this module to be
    imported in Ray worker processes, making the registry entry available.
    """

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, TimeSeriesRequest)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[0.99] * request.horizon,
        )


@register_backend("_smoke_error")
class ErrorTimeSeriesBackend(ModelBackend):
    """Backend that always raises — used to test service-boundary error handling."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        raise RuntimeError("backend exploded")
