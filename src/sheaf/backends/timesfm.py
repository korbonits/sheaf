"""TimesFM backend for time series forecasting.

Requires: pip install "sheaf-serve[time-series]"
Model: google/timesfm-1.0-200m-pytorch (CPU-compatible, ~200M params)

Key differences from Chronos-Bolt:
- horizon_len is fixed at load time; longer requests are truncated
- quantiles are always computed at all 9 levels (0.1..0.9); we select at response time
- freq is an integer: 0=high (hourly/daily), 1=medium (weekly/monthly), 2=low
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.time_series import (
    Frequency,
    OutputMode,
    TimeSeriesRequest,
    TimeSeriesResponse,
)
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

# TimesFM outputs 9 quantiles at these fixed levels
_TIMESFM_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Map our Frequency enum to TimesFM's freq integers
# 0 = high frequency (sub-daily through daily)
# 1 = medium frequency (weekly through monthly)
# 2 = low frequency (quarterly, yearly)
_FREQ_MAP: dict[Frequency, int] = {
    Frequency.MINUTELY: 0,
    Frequency.FIVE_MINUTELY: 0,
    Frequency.FIFTEEN_MINUTELY: 0,
    Frequency.HOURLY: 0,
    Frequency.DAILY: 0,
    Frequency.WEEKLY: 1,
    Frequency.MONTHLY: 1,
}


@register_backend("timesfm")
class TimesFMBackend(ModelBackend):
    """ModelBackend for Google TimesFM.

    Args:
        model_id: HuggingFace model ID. Use "google/timesfm-1.0-200m-pytorch" for CPU.
        backend: "cpu", "gpu", or "tpu"
        horizon_len: Max forecast horizon. Must match checkpoint architecture
                     (default 128). Raises if request horizon exceeds this.
        per_core_batch_size: Batch size per device.
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-1.0-200m-pytorch",
        backend: str = "cpu",
        horizon_len: int = 128,
        per_core_batch_size: int = 32,
    ) -> None:
        self._model_id = model_id
        self._backend = backend
        self._horizon_len = horizon_len
        self._per_core_batch_size = per_core_batch_size
        self._model: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def load(self) -> None:
        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "timesfm is required for the TimesFM backend. "
                "Install it with: pip install 'sheaf-serve[time-series]'"
            ) from e

        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self._backend,
                per_core_batch_size=self._per_core_batch_size,
                horizon_len=self._horizon_len,
                # Use all 9 default quantile levels — we select at response time
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self._model_id,
            ),
        )

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TimeSeriesRequest):
            raise TypeError(f"Expected TimeSeriesRequest, got {type(request)}")
        return self._run([request])[0]

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        ts_requests = [r for r in requests if isinstance(r, TimeSeriesRequest)]
        if len(ts_requests) != len(requests):
            raise TypeError("All requests must be TimeSeriesRequest for TimesFMBackend")
        return self._run(ts_requests)

    def _run(self, requests: list[TimeSeriesRequest]) -> list[TimeSeriesResponse]:
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        for req in requests:
            if req.horizon > self._horizon_len:
                raise ValueError(
                    f"Requested horizon {req.horizon} exceeds model's "
                    f"horizon_len={self._horizon_len}. "
                    f"Reinitialize with a larger horizon_len."
                )

        contexts = [np.array(r.history or [], dtype=np.float32) for r in requests]
        freqs = [_FREQ_MAP.get(requests[0].frequency, 0)] * len(requests)

        # point: [batch, horizon_len]
        # quantile_preds: [batch, horizon_len, n_quantiles=9]
        point, quantile_preds = self._model.forecast(inputs=contexts, freq=freqs)

        return [
            self._build_response(req, point[i], quantile_preds[i])
            for i, req in enumerate(requests)
        ]

    def _build_response(
        self,
        req: TimeSeriesRequest,
        point: np.ndarray,
        quantile_preds: np.ndarray,
    ) -> TimeSeriesResponse:
        h = req.horizon
        # Truncate to requested horizon
        mean = point[:h].tolist()

        quantiles = None
        if req.output_mode == OutputMode.QUANTILES:
            # quantile_preds: [horizon_len, 9] — select requested levels
            quantiles = {}
            for q in req.quantile_levels:
                if q in _TIMESFM_QUANTILE_LEVELS:
                    idx = _TIMESFM_QUANTILE_LEVELS.index(q)
                    quantiles[str(q)] = quantile_preds[:h, idx].tolist()
                else:
                    # Interpolate between nearest available quantiles
                    lower = max(lv for lv in _TIMESFM_QUANTILE_LEVELS if lv <= q)
                    upper = min(u for u in _TIMESFM_QUANTILE_LEVELS if u >= q)
                    lo_idx = _TIMESFM_QUANTILE_LEVELS.index(lower)
                    hi_idx = _TIMESFM_QUANTILE_LEVELS.index(upper)
                    weight = (q - lower) / (upper - lower) if upper != lower else 0.0
                    interp = (
                        (1 - weight) * quantile_preds[:h, lo_idx]
                        + weight * quantile_preds[:h, hi_idx]
                    )
                    quantiles[str(q)] = interp.tolist()

        return TimeSeriesResponse(
            request_id=req.request_id,
            model_name=req.model_name,
            horizon=req.horizon,
            frequency=req.frequency.value,
            mean=mean,
            quantiles=quantiles,
        )
