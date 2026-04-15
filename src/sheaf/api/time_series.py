"""API contract for time series foundation models (Chronos2, TimesFM, etc.)."""

from enum import Enum
from typing import Annotated, Any
from uuid import UUID

import numpy as np
from pydantic import Field, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class Frequency(str, Enum):
    MINUTELY = "1min"
    FIVE_MINUTELY = "5min"
    FIFTEEN_MINUTELY = "15min"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1W"
    MONTHLY = "1M"


class OutputMode(str, Enum):
    MEAN = "mean"
    QUANTILES = "quantiles"
    SAMPLES = "samples"


class TimeSeriesRequest(BaseRequest):
    """Request contract for time series foundation models.

    Either `history` (raw values) or `feature_ref` (Feast entity reference)
    must be provided, not both.
    """

    model_type: ModelType = ModelType.TIME_SERIES

    # Input: raw history or feature store reference
    history: Annotated[list[float] | None, Field(default=None)]
    feature_ref: Annotated[dict[str, str] | None, Field(default=None)]
    # e.g. {"feature_view": "asset_prices", "entity_id": "AAPL"}

    horizon: Annotated[int, Field(gt=0)]
    frequency: Frequency
    output_mode: OutputMode = OutputMode.MEAN
    quantile_levels: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Quantile levels — only used when output_mode=quantiles",
    )
    num_samples: int = Field(
        default=20,
        description="Number of samples — only used when output_mode=samples",
    )

    @model_validator(mode="after")
    def validate_input_source(self) -> "TimeSeriesRequest":
        if self.history is None and self.feature_ref is None:
            raise ValueError("One of `history` or `feature_ref` must be provided.")
        if self.history is not None and self.feature_ref is not None:
            raise ValueError("Provide either `history` or `feature_ref`, not both.")
        return self


class TimeSeriesResponse(BaseResponse):
    """Response contract for time series foundation models."""

    model_type: ModelType = ModelType.TIME_SERIES

    # Mean forecast — always populated
    mean: list[float]

    # Populated when output_mode=quantiles
    quantiles: dict[str, list[float]] | None = None
    # e.g. {"0.1": [...], "0.5": [...], "0.9": [...]}

    # Populated when output_mode=samples
    samples: list[list[float]] | None = None

    horizon: int
    frequency: str
