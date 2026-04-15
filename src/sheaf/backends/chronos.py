"""Chronos2 backend for time series forecasting.

Requires: pip install sheaf-serve[time-series]
Supports: amazon/chronos-t5-{tiny,mini,small,base,large}
          amazon/chronos-bolt-{tiny,mini,small,base}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.time_series import OutputMode, TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

if TYPE_CHECKING:
    import torch


@register_backend("chronos2")
class Chronos2Backend(ModelBackend):
    """ModelBackend implementation for Chronos / Chronos-Bolt models.

    Args:
        model_id: HuggingFace model ID, e.g. "amazon/chronos-bolt-small"
        device_map: "cpu", "cuda", "mps", or "auto"
        torch_dtype: "bfloat16", "float32", etc. Passed to from_pretrained.
        num_samples: Default number of samples for probabilistic output.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-small",
        device_map: str = "cpu",
        torch_dtype: str = "bfloat16",
        num_samples: int = 20,
    ) -> None:
        self._model_id = model_id
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._default_num_samples = num_samples
        self._pipeline: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def load(self) -> None:
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting is required for the Chronos2 backend. "
                "Install it with: pip install sheaf-serve[time-series]"
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(self._torch_dtype, torch.bfloat16)

        self._pipeline = BaseChronosPipeline.from_pretrained(
            self._model_id,
            device_map=self._device_map,
            torch_dtype=torch_dtype,
        )

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TimeSeriesRequest):
            raise TypeError(f"Expected TimeSeriesRequest, got {type(request)}")
        return self._run([request])[0]

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        ts_requests = [r for r in requests if isinstance(r, TimeSeriesRequest)]
        if len(ts_requests) != len(requests):
            raise TypeError("All requests must be TimeSeriesRequest for Chronos2Backend")
        return self._run(ts_requests)

    def _run(self, requests: list[TimeSeriesRequest]) -> list[TimeSeriesResponse]:
        import torch

        if self._pipeline is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        # Build context tensors — pad to same length within batch
        histories = [r.history or [] for r in requests]
        max_len = max(len(h) for h in histories)
        padded = [
            [float("nan")] * (max_len - len(h)) + h
            for h in histories
        ]
        context = torch.tensor(padded, dtype=torch.float32)

        # Bucket by horizon: all requests in a batch must share a horizon
        # (caller is responsible for bucketing — see BatchPolicy(bucket_by="horizon"))
        horizon = requests[0].horizon
        num_samples = requests[0].num_samples

        # predict returns [batch, num_samples, horizon]
        forecast = self._pipeline.predict(
            context=context,
            prediction_length=horizon,
            num_samples=num_samples,
        )
        forecast_np: np.ndarray = forecast.numpy()

        responses = []
        for i, req in enumerate(requests):
            samples = forecast_np[i]  # [num_samples, horizon]
            responses.append(self._build_response(req, samples))
        return responses

    def _build_response(
        self, req: TimeSeriesRequest, samples: np.ndarray
    ) -> TimeSeriesResponse:
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
