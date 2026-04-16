"""Tests for API contracts."""

import pytest
from pydantic import ValidationError

from sheaf.api.time_series import FeatureRef, Frequency, OutputMode, TimeSeriesRequest


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
    ref = FeatureRef(
        feature_view="asset_prices",
        feature_name="close_history_30d",
        entity_key="ticker",
        entity_value="AAPL",
    )
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        feature_ref=ref,
        horizon=24,
        frequency=Frequency.DAILY,
        output_mode=OutputMode.QUANTILES,
    )
    assert req.history is None
    assert req.feature_ref == ref
    assert req.quantile_levels == [0.1, 0.5, 0.9]


def test_time_series_request_feature_ref_accepts_dict():
    """feature_ref can be passed as a plain dict — Pydantic coerces it."""
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        feature_ref={
            "feature_view": "asset_prices",
            "feature_name": "close_history_30d",
            "entity_key": "ticker",
            "entity_value": "AAPL",
        },
        horizon=24,
        frequency=Frequency.DAILY,
    )
    assert isinstance(req.feature_ref, FeatureRef)
    assert req.feature_ref.entity_value == "AAPL"


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
            feature_ref=FeatureRef(
                feature_view="asset_prices",
                feature_name="close_history_30d",
                entity_key="ticker",
                entity_value="AAPL",
            ),
            horizon=12,
            frequency=Frequency.HOURLY,
        )


# ---------------------------------------------------------------------------
# Multivariate history
# ---------------------------------------------------------------------------


def test_time_series_request_multivariate_history():
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        history=[[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
        horizon=6,
        frequency=Frequency.HOURLY,
    )
    assert req.n_variates == 2
    assert req.target_index == 0
    assert req.target_history == [1.0, 2.0, 3.0]


def test_time_series_request_multivariate_target_index():
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        history=[[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]],
        horizon=6,
        frequency=Frequency.HOURLY,
        target_index=1,
    )
    assert req.n_variates == 2
    assert req.target_history == [10.0, 11.0, 12.0]


def test_time_series_request_univariate_target_history():
    req = TimeSeriesRequest(
        model_name="chronos2-small",
        history=[1.0, 2.0, 3.0],
        horizon=6,
        frequency=Frequency.HOURLY,
    )
    assert req.n_variates == 1
    assert req.target_history == [1.0, 2.0, 3.0]


def test_time_series_request_rejects_jagged_multivariate():
    with pytest.raises(ValidationError, match="same length"):
        TimeSeriesRequest(
            model_name="chronos2-small",
            history=[[1.0, 2.0], [3.0]],  # jagged
            horizon=6,
            frequency=Frequency.HOURLY,
        )


def test_time_series_request_rejects_out_of_range_target_index():
    with pytest.raises(ValidationError, match="target_index"):
        TimeSeriesRequest(
            model_name="chronos2-small",
            history=[[1.0, 2.0], [3.0, 4.0]],
            horizon=6,
            frequency=Frequency.HOURLY,
            target_index=5,  # only 2 variates
        )
