"""Tests for the serving layer.

Full Ray Serve integration tests (ModelServer.run) require a running Ray
cluster and are out of scope for the unit test suite.  These tests cover:

  - ModelBackend.async_predict / async_batch_predict (thread-executor path)
  - AnyRequest discriminated-union parsing and 422 validation
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.tabular import TabularRequest
from sheaf.api.time_series import Frequency, TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.server import AnyRequest

# ---------------------------------------------------------------------------
# Stub backend
# ---------------------------------------------------------------------------


class _StubTimeSeriesBackend(ModelBackend):
    """Minimal synchronous backend used to test async dispatch."""

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
            mean=[0.5] * request.horizon,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> _StubTimeSeriesBackend:
    b = _StubTimeSeriesBackend()
    b.load()
    return b


@pytest.fixture
def ts_request() -> TimeSeriesRequest:
    return TimeSeriesRequest(
        model_name="stub",
        history=[1.0, 2.0, 3.0, 4.0],
        horizon=6,
        frequency=Frequency.HOURLY,
    )


# ---------------------------------------------------------------------------
# async_predict / async_batch_predict
# ---------------------------------------------------------------------------


async def test_async_predict_returns_correct_response(
    backend: _StubTimeSeriesBackend,
    ts_request: TimeSeriesRequest,
) -> None:
    response = await backend.async_predict(ts_request)
    assert response.request_id == ts_request.request_id
    assert isinstance(response, TimeSeriesResponse)
    assert response.mean == [0.5] * 6


async def test_async_batch_predict_preserves_order(
    backend: _StubTimeSeriesBackend,
    ts_request: TimeSeriesRequest,
) -> None:
    req2 = ts_request.model_copy(update={"horizon": 3})
    responses = await backend.async_batch_predict([ts_request, req2])
    assert len(responses) == 2
    assert responses[0].request_id == ts_request.request_id
    assert responses[1].request_id == req2.request_id
    assert len(responses[0].mean) == 6  # type: ignore[union-attr]
    assert len(responses[1].mean) == 3  # type: ignore[union-attr]


async def test_async_predict_runs_in_executor(
    backend: _StubTimeSeriesBackend,
    ts_request: TimeSeriesRequest,
) -> None:
    """Smoke-test: async_predict completes without blocking the event loop."""
    import asyncio

    results = await asyncio.gather(
        backend.async_predict(ts_request),
        backend.async_predict(ts_request),
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# AnyRequest discriminated union
# ---------------------------------------------------------------------------

_adapter: TypeAdapter[AnyRequest] = TypeAdapter(AnyRequest)  # type: ignore[valid-type]


def test_any_request_parses_time_series() -> None:
    req = _adapter.validate_python(
        {
            "model_type": "time_series",
            "model_name": "chronos",
            "history": [1.0, 2.0, 3.0],
            "horizon": 4,
            "frequency": "1h",
        }
    )
    assert isinstance(req, TimeSeriesRequest)
    assert req.horizon == 4


def test_any_request_parses_tabular() -> None:
    req = _adapter.validate_python(
        {
            "model_type": "tabular",
            "model_name": "tabpfn",
            "context_X": [[1.0, 2.0], [3.0, 4.0]],
            "context_y": [0, 1],
            "query_X": [[5.0, 6.0]],
        }
    )
    assert isinstance(req, TabularRequest)


def test_any_request_rejects_unknown_model_type() -> None:
    with pytest.raises(ValidationError):
        _adapter.validate_python(
            {
                "model_type": "molecular",  # not yet in union
                "model_name": "esm3",
            }
        )


def test_any_request_rejects_invalid_time_series_body() -> None:
    with pytest.raises(ValidationError):
        _adapter.validate_python(
            {
                "model_type": "time_series",
                "model_name": "chronos",
                # missing horizon and frequency
            }
        )
