"""Moirai backend for time series forecasting.

Requires: pip install "sheaf-serve[moirai]"
Model family: Salesforce/moirai-{1.0,1.1}-R-{small,base,large}

Key differences from Chronos / TimesFM:
- MoiraiModule is loaded once; a lightweight MoiraiForecast wrapper is created
  per prediction group (fixed prediction_length required by the API).
- Multivariate-native: all variates are passed as context; only the variate at
  target_index is returned in the response.
- Uses GluonTS ListDataset as the input interface.
- context_length caps how many trailing history steps are used; longer histories
  are trimmed from the left.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

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

# Map our Frequency enum to pandas period freq strings
_FREQ_MAP: dict[Frequency, str] = {
    Frequency.MINUTELY: "min",
    Frequency.FIVE_MINUTELY: "5min",
    Frequency.FIFTEEN_MINUTELY: "15min",
    Frequency.HOURLY: "h",
    Frequency.DAILY: "D",
    Frequency.WEEKLY: "W",
    Frequency.MONTHLY: "ME",
}


@register_backend("moirai")
class MoiraiBackend(ModelBackend):
    """ModelBackend for Salesforce Moirai (uni2ts).

    Args:
        model_id: HuggingFace model ID.
            Moirai 1.1 (recommended): "Salesforce/moirai-1.1-R-small" through "-large"
            Moirai 1.0: "Salesforce/moirai-1.0-R-small" through "-large"
        context_length: Trailing history steps passed as context (longer histories
            are trimmed from the left). Default 1000.
        patch_size: Patch size for the transformer. "auto" lets the model choose.
        num_samples: Number of forecast samples for probabilistic output.
    """

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-1.1-R-small",
        context_length: int = 1000,
        patch_size: str | int = "auto",
        num_samples: int = 100,
    ) -> None:
        self._model_id = model_id
        self._context_length = context_length
        self._patch_size = patch_size
        self._num_samples = num_samples
        self._module: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def load(self) -> None:
        try:
            from uni2ts.model.moirai import (  # ty: ignore[unresolved-import]
                MoiraiModule,
            )
        except ImportError as e:
            raise ImportError(
                "uni2ts is required for the Moirai backend. "
                "Install it with: pip install 'sheaf-serve[moirai]'"
            ) from e
        self._module = MoiraiModule.from_pretrained(self._model_id)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TimeSeriesRequest):
            raise TypeError(f"Expected TimeSeriesRequest, got {type(request)}")
        return self._run([request])[0]

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        ts_requests = [r for r in requests if isinstance(r, TimeSeriesRequest)]
        if len(ts_requests) != len(requests):
            raise TypeError("All requests must be TimeSeriesRequest for MoiraiBackend")
        return cast("list[BaseResponse]", self._run(ts_requests))

    def _run(self, requests: list[TimeSeriesRequest]) -> list[TimeSeriesResponse]:
        import pandas as pd  # ty: ignore[unresolved-import]
        from gluonts.dataset.common import (  # ty: ignore[unresolved-import]
            ListDataset,
        )
        from uni2ts.model.moirai import (  # ty: ignore[unresolved-import]
            MoiraiForecast,
        )

        if self._module is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        # Group by (horizon, n_variates) so each group shares one predictor.
        # Ray Serve may batch requests with different horizons together.
        groups: dict[tuple[int, int], list[tuple[int, TimeSeriesRequest]]] = (
            defaultdict(list)
        )
        for idx, req in enumerate(requests):
            groups[(req.horizon, req.n_variates)].append((idx, req))

        responses: list[TimeSeriesResponse | None] = [None] * len(requests)

        for (horizon, n_variates), group in groups.items():
            freq = _FREQ_MAP.get(group[0][1].frequency, "h")

            predictor = MoiraiForecast(
                module=self._module,
                prediction_length=horizon,
                context_length=self._context_length,
                target_dim=n_variates,
                feat_dynamic_real_dim=0,
                patch_size=self._patch_size,
                num_samples=self._num_samples,
            ).create_predictor(batch_size=len(group))

            items = [
                {
                    "target": self._build_target(req, n_variates),
                    "start": pd.Period(pd.Timestamp("2020-01-01"), freq=freq),
                }
                for _, req in group
            ]
            dataset = ListDataset(items, freq=freq, one_dim_target=(n_variates == 1))
            forecasts = list(predictor.predict(dataset))

            for (orig_idx, req), forecast in zip(group, forecasts):
                responses[orig_idx] = self._build_response(req, forecast, n_variates)

        return cast("list[TimeSeriesResponse]", responses)

    def _build_target(self, req: TimeSeriesRequest, n_variates: int) -> np.ndarray:
        """Build target array in GluonTS convention, trimmed to context_length.

        Univariate:   shape [time]
        Multivariate: shape [variates, time]  (GluonTS transposes vs. our API)
        """
        if n_variates == 1:
            arr = np.array(req.target_history, dtype=np.float32)
            return arr[-self._context_length :]

        # req.history is list[list[float]], shape [time, variates]
        arr = np.array(req.history, dtype=np.float32)  # [time, variates]
        arr = arr[-self._context_length :]  # trim from left
        return arr.T  # [variates, time]

    def _build_response(
        self,
        req: TimeSeriesRequest,
        forecast: Any,
        n_variates: int,
    ) -> TimeSeriesResponse:
        h = req.horizon

        # forecast.quantile(q) returns:
        #   univariate:   [horizon]
        #   multivariate: [horizon, variates]
        median = forecast.quantile(0.5)
        if n_variates > 1:
            mean: list[float] = median[:h, req.target_index].tolist()
        else:
            mean = median[:h].tolist()

        quantiles = None
        if req.output_mode == OutputMode.QUANTILES:
            quantiles = {}
            for q in req.quantile_levels:
                q_arr = forecast.quantile(q)
                if n_variates > 1:
                    quantiles[str(q)] = q_arr[:h, req.target_index].tolist()
                else:
                    quantiles[str(q)] = q_arr[:h].tolist()

        raw_samples = None
        if req.output_mode == OutputMode.SAMPLES:
            # forecast.samples shape:
            #   univariate:   [num_samples, horizon]
            #   multivariate: [num_samples, horizon, variates]
            s = forecast.samples
            if n_variates > 1:
                raw_samples = s[:, :h, req.target_index].tolist()
            else:
                raw_samples = s[:, :h].tolist()

        return TimeSeriesResponse(
            request_id=req.request_id,
            model_name=req.model_name,
            horizon=req.horizon,
            frequency=req.frequency.value,
            mean=mean,
            quantiles=quantiles,
            samples=raw_samples,
        )
