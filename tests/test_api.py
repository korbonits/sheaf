"""Tests for API contracts."""

import pytest
from pydantic import ValidationError

from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest


def test_time_series_request_with_raw_history():
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        history=[1.0, 2.0, 3.0, 4.0, 5.0],
        horizon=12,
        frequency=Frequency.HOURLY,
    )
    assert req.horizon == 12
    assert req.output_mode == OutputMode.MEAN
    assert req.feature_ref is None


def test_time_series_request_with_feature_ref():
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        feature_ref={"feature_view": "asset_prices", "entity_id": "AAPL"},
        horizon=24,
        frequency=Frequency.DAILY,
        output_mode=OutputMode.QUANTILES,
    )
    assert req.history is None
    assert req.quantile_levels == [0.1, 0.5, 0.9]


def test_time_series_request_requires_input_source():
    with pytest.raises(ValidationError, match="history.*feature_ref"):
        TimeSeriesRequest(
            model_name="chronos2-small",
            horizon=12,
            frequency=Frequency.HOURLY,
        )


def test_time_series_request_rejects_both_input_sources():
    with pytest.raises(ValidationError, match="either"):
        TimeSeriesRequest(
            model_name="chronos2-small",
            history=[1.0, 2.0, 3.0],
            feature_ref={"feature_view": "asset_prices", "entity_id": "AAPL"},
            horizon=12,
            frequency=Frequency.HOURLY,
        )
