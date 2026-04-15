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


class _ErrorBackend(ModelBackend):
    """Backend that always raises — mirrors tests.stubs.ErrorTimeSeriesBackend."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        raise RuntimeError("backend exploded")


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


# ---------------------------------------------------------------------------
# Registry auto-import fix
# ---------------------------------------------------------------------------


def test_standard_backends_registered_after_server_import() -> None:
    """Importing sheaf.server must register all standard backends.

    This directly tests the auto-import fix: Ray worker processes import
    server.py on startup, so @register_backend must have run for each
    standard backend by the time any deployment __init__ is called.
    """
    from sheaf.registry import _BACKEND_REGISTRY

    assert "chronos2" in _BACKEND_REGISTRY, (
        "chronos2 not in registry — sheaf.backends.chronos was not auto-imported"
    )
    assert "tabpfn" in _BACKEND_REGISTRY, (
        "tabpfn not in registry — sheaf.backends.tabpfn was not auto-imported"
    )
    assert "timesfm" in _BACKEND_REGISTRY, (
        "timesfm not in registry — sheaf.backends.timesfm was not auto-imported"
    )


# ---------------------------------------------------------------------------
# BatchPolicy wiring
# ---------------------------------------------------------------------------


def test_batch_policy_defaults() -> None:
    """BatchPolicy defaults match the @serve.batch decorator defaults."""
    from sheaf.scheduling.batch import BatchPolicy

    policy = BatchPolicy()
    assert policy.max_batch_size == 32
    assert policy.timeout_ms == 50


def test_batch_policy_custom_values_are_stored() -> None:
    """ModelSpec stores custom BatchPolicy and exposes it for __init__ wiring."""
    from sheaf.api.base import ModelType
    from sheaf.scheduling.batch import BatchPolicy
    from sheaf.spec import ModelSpec

    spec = ModelSpec(
        name="test",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        batch_policy=BatchPolicy(max_batch_size=8, timeout_ms=20),
    )
    assert spec.batch_policy.max_batch_size == 8
    assert spec.batch_policy.timeout_ms == 20


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.fixture
def error_backend() -> _ErrorBackend:
    b = _ErrorBackend()
    b.load()
    return b


async def test_async_predict_propagates_backend_exception(
    error_backend: _ErrorBackend,
    ts_request: TimeSeriesRequest,
) -> None:
    """async_predict must surface exceptions raised by the sync predict()."""
    with pytest.raises(RuntimeError, match="backend exploded"):
        await error_backend.async_predict(ts_request)


async def test_async_batch_predict_propagates_backend_exception(
    error_backend: _ErrorBackend,
    ts_request: TimeSeriesRequest,
) -> None:
    """async_batch_predict must surface exceptions raised by batch_predict()."""
    with pytest.raises(RuntimeError, match="backend exploded"):
        await error_backend.async_batch_predict([ts_request])


# ---------------------------------------------------------------------------
# ModelServer.update — hot-swap
# ---------------------------------------------------------------------------


def test_update_raises_for_unknown_deployment() -> None:
    """update() must raise ValueError when the deployment name doesn't exist."""
    from sheaf.api.base import ModelType
    from sheaf.server import ModelServer
    from sheaf.spec import ModelSpec

    server = ModelServer(models=[])
    spec = ModelSpec(
        name="does-not-exist",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
    )
    with pytest.raises(ValueError, match="does-not-exist"):
        server.update(spec)
