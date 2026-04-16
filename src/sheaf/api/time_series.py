"""API contract for time series foundation models (Chronos2, TimesFM, etc.)."""

from enum import Enum
from typing import Annotated, Literal, cast

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

    Univariate:   history=[1.0, 2.0, 3.0, ...]
    Multivariate: history=[[1.0, 10.0], [2.0, 11.0], ...]  (shape: [time, variates])
                  target_index selects which variate to forecast (default 0).
    """

    model_type: Literal[ModelType.TIME_SERIES] = ModelType.TIME_SERIES

    # Input: raw history or feature store reference
    history: Annotated[list[float] | list[list[float]] | None, Field(default=None)]
    feature_ref: Annotated[dict[str, str] | None, Field(default=None)]
    # e.g. {"feature_view": "asset_prices", "entity_id": "AAPL"}

    # Multivariate: index of the variate to forecast (0-based)
    target_index: int = Field(
        default=0,
        ge=0,
        description="Variate index to forecast — only used for multivariate history",
    )

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
        if self.history is not None and len(self.history) > 0:
            first = self.history[0]
            if isinstance(first, list):
                # Multivariate: validate uniform shape and target_index range
                n = len(first)
                for row in self.history:
                    if not isinstance(row, list) or len(row) != n:  # type: ignore[arg-type]
                        raise ValueError(
                            "All rows in a multivariate `history` must have "
                            "the same length."
                        )
                if self.target_index >= n:
                    raise ValueError(
                        f"`target_index` {self.target_index} is out of range "
                        f"for history with {n} variates (valid range: 0–{n - 1})."
                    )
        return self

    @property
    def n_variates(self) -> int:
        """Number of variates in history (1 for univariate)."""
        if not self.history:
            return 1
        return len(self.history[0]) if isinstance(self.history[0], list) else 1

    @property
    def target_history(self) -> list[float]:
        """Univariate target series extracted from (possibly multivariate) history.

        For univariate history, returns history as-is.
        For multivariate history (shape [time, variates]), extracts the variate
        at target_index.
        """
        if not self.history:
            return []
        if isinstance(self.history[0], list):
            # history is list[list[float]] — extract target column
            mv = cast("list[list[float]]", self.history)
            return [row[self.target_index] for row in mv]
        return cast("list[float]", self.history)


class TimeSeriesResponse(BaseResponse):
    """Response contract for time series foundation models."""

    model_type: Literal[ModelType.TIME_SERIES] = ModelType.TIME_SERIES

    # Mean forecast — always populated
    mean: list[float]

    # Populated when output_mode=quantiles
    quantiles: dict[str, list[float]] | None = None
    # e.g. {"0.1": [...], "0.5": [...], "0.9": [...]}

    # Populated when output_mode=samples
    samples: list[list[float]] | None = None

    horizon: int
    frequency: str
