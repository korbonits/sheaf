"""Tests for MoiraiBackend — fully mocked, no real uni2ts install required.

Covers:
  - load() raises ImportError when uni2ts is absent
  - load() succeeds when uni2ts is present (module stub)
  - predict() univariate mean output
  - predict() univariate quantiles output
  - predict() univariate samples output
  - predict() multivariate history — all variates passed as context,
    target variate extracted from forecast
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.time_series import Frequency, TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.moirai import MoiraiBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HORIZON = 4
_NUM_SAMPLES = 10


def _make_forecast(horizon: int, n_variates: int = 1) -> MagicMock:
    """Stub GluonTS forecast object."""
    forecast = MagicMock()

    if n_variates == 1:
        # quantile(q) → [horizon]
        forecast.quantile.side_effect = lambda q: np.full(horizon, float(q))
        # samples → [num_samples, horizon]
        forecast.samples = np.ones((_NUM_SAMPLES, horizon)) * 0.5
    else:
        # quantile(q) → [horizon, variates]
        forecast.quantile.side_effect = lambda q: np.full(
            (horizon, n_variates), float(q)
        )
        # samples → [num_samples, horizon, variates]
        forecast.samples = np.ones((_NUM_SAMPLES, horizon, n_variates)) * 0.5

    return forecast


def _make_uni2ts_mod(horizon: int = _HORIZON, n_variates: int = 1) -> ModuleType:
    """Fake uni2ts.model.moirai with stub MoiraiModule and MoiraiForecast."""
    mod = ModuleType("uni2ts.model.moirai")

    forecast = _make_forecast(horizon, n_variates)
    predictor = MagicMock()
    predictor.predict.return_value = [forecast]

    forecast_cls = MagicMock()
    forecast_instance = MagicMock()
    forecast_instance.create_predictor.return_value = predictor
    forecast_cls.return_value = forecast_instance

    module_cls = MagicMock()
    mod.MoiraiModule = module_cls
    mod.MoiraiForecast = forecast_cls
    return mod


def _make_gluonts_mod() -> ModuleType:
    """Fake gluonts.dataset.common with a passthrough ListDataset."""
    mod = ModuleType("gluonts.dataset.common")
    mod.ListDataset = MagicMock(return_value=MagicMock())
    return mod


def _make_pandas_mod() -> ModuleType:
    """Minimal pandas stub — only Period and Timestamp are needed."""
    import types

    mod = types.ModuleType("pandas")

    class _Period:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class _Timestamp:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    mod.Period = _Period  # type: ignore[attr-defined]
    mod.Timestamp = _Timestamp  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_backend() -> MoiraiBackend:
    """MoiraiBackend with load() already called (uni2ts mocked)."""
    backend = MoiraiBackend(num_samples=_NUM_SAMPLES)
    uni2ts_mod = _make_uni2ts_mod()
    uni2ts_pkg = ModuleType("uni2ts")

    with patch.dict(
        sys.modules,
        {
            "uni2ts": uni2ts_pkg,
            "uni2ts.model": ModuleType("uni2ts.model"),
            "uni2ts.model.moirai": uni2ts_mod,
        },
    ):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_uni2ts() -> None:
    """load() raises ImportError when uni2ts is not installed."""
    backend = MoiraiBackend()

    def _raise_on_uni2ts(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "uni2ts.model.moirai":
            raise ModuleNotFoundError("No module named 'uni2ts'")
        return __import__(name, *args, **kwargs)

    mods_without = {k: v for k, v in sys.modules.items() if "uni2ts" not in k}
    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise_on_uni2ts),
        pytest.raises(ImportError, match="sheaf-serve\\[moirai\\]"),
    ):
        backend.load()


def test_load_succeeds_with_uni2ts() -> None:
    """load() completes without error when uni2ts is available (mocked)."""
    backend = MoiraiBackend()
    uni2ts_mod = _make_uni2ts_mod()
    uni2ts_pkg = ModuleType("uni2ts")

    with patch.dict(
        sys.modules,
        {
            "uni2ts": uni2ts_pkg,
            "uni2ts.model": ModuleType("uni2ts.model"),
            "uni2ts.model.moirai": uni2ts_mod,
        },
    ):
        backend.load()

    assert backend._module is not None


# ---------------------------------------------------------------------------
# Univariate predict
# ---------------------------------------------------------------------------


def _uv_request(**kwargs) -> TimeSeriesRequest:  # type: ignore[no-untyped-def]
    return TimeSeriesRequest(
        model_name="moirai",
        history=[1.0, 2.0, 3.0, 4.0],
        horizon=_HORIZON,
        frequency=Frequency.HOURLY,
        **kwargs,
    )


def test_predict_univariate_mean(loaded_backend: MoiraiBackend) -> None:
    uni2ts_mod = _make_uni2ts_mod(horizon=_HORIZON, n_variates=1)
    gluonts_mod = _make_gluonts_mod()
    pandas_mod = _make_pandas_mod()

    with patch.dict(
        sys.modules,
        {
            "pandas": pandas_mod,
            "uni2ts.model.moirai": uni2ts_mod,
            "gluonts.dataset.common": gluonts_mod,
        },
    ):
        resp = loaded_backend.predict(_uv_request())

    assert isinstance(resp, TimeSeriesResponse)
    assert resp.model_type == "time_series"
    assert len(resp.mean) == _HORIZON
    # quantile(0.5) returns 0.5 for all steps in the mock
    assert resp.mean == [0.5] * _HORIZON
    assert resp.quantiles is None
    assert resp.samples is None


def test_predict_univariate_quantiles(loaded_backend: MoiraiBackend) -> None:
    from sheaf.api.time_series import OutputMode

    uni2ts_mod = _make_uni2ts_mod(horizon=_HORIZON, n_variates=1)
    gluonts_mod = _make_gluonts_mod()
    pandas_mod = _make_pandas_mod()

    with patch.dict(
        sys.modules,
        {
            "pandas": pandas_mod,
            "uni2ts.model.moirai": uni2ts_mod,
            "gluonts.dataset.common": gluonts_mod,
        },
    ):
        resp = loaded_backend.predict(
            _uv_request(
                output_mode=OutputMode.QUANTILES,
                quantile_levels=[0.1, 0.5, 0.9],
            )
        )

    assert resp.quantiles is not None
    assert set(resp.quantiles.keys()) == {"0.1", "0.5", "0.9"}
    assert resp.quantiles["0.1"] == [0.1] * _HORIZON
    assert resp.quantiles["0.9"] == [0.9] * _HORIZON


def test_predict_univariate_samples(loaded_backend: MoiraiBackend) -> None:
    from sheaf.api.time_series import OutputMode

    uni2ts_mod = _make_uni2ts_mod(horizon=_HORIZON, n_variates=1)
    gluonts_mod = _make_gluonts_mod()
    pandas_mod = _make_pandas_mod()

    with patch.dict(
        sys.modules,
        {
            "pandas": pandas_mod,
            "uni2ts.model.moirai": uni2ts_mod,
            "gluonts.dataset.common": gluonts_mod,
        },
    ):
        resp = loaded_backend.predict(_uv_request(output_mode=OutputMode.SAMPLES))

    assert resp.samples is not None
    assert len(resp.samples) == _NUM_SAMPLES
    assert len(resp.samples[0]) == _HORIZON


# ---------------------------------------------------------------------------
# Multivariate predict
# ---------------------------------------------------------------------------


def test_predict_multivariate_uses_target_index(
    loaded_backend: MoiraiBackend,
) -> None:
    """All variates passed as context; only target_index variate is returned."""
    # 2 variates: variate 0 = [1,2,3,4], variate 1 = [10,20,30,40]
    req = TimeSeriesRequest(
        model_name="moirai",
        history=[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        horizon=_HORIZON,
        frequency=Frequency.HOURLY,
        target_index=1,  # forecast variate 1
    )

    uni2ts_mod = _make_uni2ts_mod(horizon=_HORIZON, n_variates=2)
    gluonts_mod = _make_gluonts_mod()
    pandas_mod = _make_pandas_mod()

    with patch.dict(
        sys.modules,
        {
            "pandas": pandas_mod,
            "uni2ts.model.moirai": uni2ts_mod,
            "gluonts.dataset.common": gluonts_mod,
        },
    ):
        resp = loaded_backend.predict(req)

    assert isinstance(resp, TimeSeriesResponse)
    assert len(resp.mean) == _HORIZON
    # quantile(0.5) → 0.5 at all positions for all variates in mock
    assert resp.mean == [0.5] * _HORIZON


def test_predict_multivariate_target_dim_passed_to_forecast(
    loaded_backend: MoiraiBackend,
) -> None:
    """MoiraiForecast must be constructed with target_dim=n_variates."""
    req = TimeSeriesRequest(
        model_name="moirai",
        history=[[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]],
        horizon=_HORIZON,
        frequency=Frequency.DAILY,
    )

    uni2ts_mod = _make_uni2ts_mod(horizon=_HORIZON, n_variates=3)
    gluonts_mod = _make_gluonts_mod()
    pandas_mod = _make_pandas_mod()

    with patch.dict(
        sys.modules,
        {
            "pandas": pandas_mod,
            "uni2ts.model.moirai": uni2ts_mod,
            "gluonts.dataset.common": gluonts_mod,
        },
    ):
        loaded_backend.predict(req)

    # Verify MoiraiForecast was constructed with target_dim=3
    forecast_cls = uni2ts_mod.MoiraiForecast
    call_kwargs = forecast_cls.call_args.kwargs
    assert call_kwargs["target_dim"] == 3
