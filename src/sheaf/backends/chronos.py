"""Chronos backend for time series forecasting.

Requires: pip install "sheaf-serve[time-series]"

Supports two model families with different inference patterns:
- ChronosBoltPipeline: returns quantiles directly [batch, 9, horizon]
  Models: amazon/chronos-bolt-{tiny,mini,small,base}
- Chronos2Pipeline: returns samples [batch, num_samples, horizon]
  Models: amazon/chronos-t5-{tiny,mini,small,base,large}
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.time_series import OutputMode, TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

# Fixed quantile levels output by all official Chronos-Bolt models
_BOLT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@register_backend("chronos2")
class Chronos2Backend(ModelBackend):
    """ModelBackend for Chronos-Bolt and Chronos2 models.

    Args:
        model_id: HuggingFace model ID.
            Bolt (fast, CPU-friendly): "amazon/chronos-bolt-tiny" through "...-base"
            Chronos2 (probabilistic): "amazon/chronos-t5-tiny" through "...-large"
        device_map: "cpu", "cuda", "mps", or "auto"
        torch_dtype: "bfloat16" or "float32". Use "float32" on CPU.
        num_samples: Samples to draw (Chronos2 only — Bolt ignores this).
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-tiny",
        device_map: str = "cpu",
        torch_dtype: str = "float32",
        num_samples: int = 20,
    ) -> None:
        self._model_id = model_id
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._default_num_samples = num_samples
        self._pipeline: Any = None
        self._is_bolt: bool = False

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def load(self) -> None:
        try:
            import torch  # ty: ignore[unresolved-import]
            from chronos import (  # ty: ignore[unresolved-import]
                BaseChronosPipeline,
                ChronosBoltPipeline,
            )
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting is required for the Chronos2 backend. "
                "Install it with: pip install 'sheaf-serve[time-series]'"
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(self._torch_dtype, torch.float32)

        self._pipeline = BaseChronosPipeline.from_pretrained(
            self._model_id,
            device_map=self._device_map,
            dtype=torch_dtype,
        )
        self._is_bolt = isinstance(self._pipeline, ChronosBoltPipeline)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TimeSeriesRequest):
            raise TypeError(f"Expected TimeSeriesRequest, got {type(request)}")
        return self._run([request])[0]

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        ts_requests = [r for r in requests if isinstance(r, TimeSeriesRequest)]
        if len(ts_requests) != len(requests):
            raise TypeError(
                "All requests must be TimeSeriesRequest for Chronos2Backend"
            )
        return cast("list[BaseResponse]", self._run(ts_requests))

    def _run(self, requests: list[TimeSeriesRequest]) -> list[TimeSeriesResponse]:
        import torch  # ty: ignore[unresolved-import]

        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        # Build context — list of tensors (handles variable-length history)
        contexts = [
            torch.tensor(r.history or [], dtype=torch.float32) for r in requests
        ]
        horizon = requests[0].horizon

        if self._is_bolt:
            # Returns [batch, num_quantiles=9, horizon]
            forecast = self._pipeline.predict(
                inputs=contexts,
                prediction_length=horizon,
            )
            forecast_np: np.ndarray = forecast.numpy()
            return [
                self._build_bolt_response(req, forecast_np[i])
                for i, req in enumerate(requests)
            ]
        else:
            # Returns list of [num_samples, horizon] tensors
            forecasts = self._pipeline.predict(
                inputs=contexts,
                prediction_length=horizon,
                num_samples=requests[0].num_samples,
            )
            return [
                self._build_sample_response(req, forecasts[i].numpy())
                for i, req in enumerate(requests)
            ]

    def _build_bolt_response(
        self, req: TimeSeriesRequest, quantile_forecast: np.ndarray
    ) -> TimeSeriesResponse:
        # quantile_forecast: [9, horizon] at levels [0.1, ..., 0.9]
        median_idx = _BOLT_QUANTILE_LEVELS.index(0.5)
        mean = quantile_forecast[median_idx].tolist()

        quantiles = None
        if req.output_mode == OutputMode.QUANTILES:
            available = {
                str(q): quantile_forecast[i].tolist()
                for i, q in enumerate(_BOLT_QUANTILE_LEVELS)
            }
            # Return requested levels; fall back to nearest available
            quantiles = {
                str(q): available.get(str(q), quantile_forecast[median_idx].tolist())
                for q in req.quantile_levels
            }

        return TimeSeriesResponse(
            request_id=req.request_id,
            model_name=req.model_name,
            horizon=req.horizon,
            frequency=req.frequency.value,
            mean=mean,
            quantiles=quantiles,
        )

    def _build_sample_response(
        self, req: TimeSeriesRequest, samples: np.ndarray
    ) -> TimeSeriesResponse:
        # samples: [num_samples, horizon]
        mean = samples.mean(axis=0).tolist()

        quantiles = None
        if req.output_mode == OutputMode.QUANTILES:
            quantiles = {
                str(q): np.quantile(samples, q, axis=0).tolist()
                for q in req.quantile_levels
            }

        raw_samples = None
        if req.output_mode == OutputMode.SAMPLES:
            raw_samples = samples.tolist()

        return TimeSeriesResponse(
            request_id=req.request_id,
            model_name=req.model_name,
            horizon=req.horizon,
            frequency=req.frequency.value,
            mean=mean,
            quantiles=quantiles,
            samples=raw_samples,
        )
