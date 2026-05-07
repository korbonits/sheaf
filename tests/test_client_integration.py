"""Integration tests: AsyncSheafClient against the real ``_build_asgi_app``.

Uses ``httpx.ASGITransport`` to drive the FastAPI app in-process — no real HTTP
server, no Ray cluster, but every other layer of the stack is real (FastAPI
routing, AnyRequest / AnyResponse pydantic discrimination, the smoke stub
backends from tests.stubs, the actual error mapping in ``modal_server.py``).

ASGITransport is async-only, so these tests use AsyncSheafClient.  Sync
client code is covered by ``test_client.py`` with ``httpx.MockTransport``;
the underlying request/decode/error-mapping helpers are shared, so async
coverage here also exercises the sync paths.
"""

from __future__ import annotations

import httpx
import pytest

import tests.stubs  # noqa: F401 — registers smoke backends
from sheaf.api.base import ModelType
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.api.tabular import TabularRequest, TabularResponse
from sheaf.api.time_series import (
    Frequency,
    TimeSeriesRequest,
    TimeSeriesResponse,
)
from sheaf.client import (
    AsyncSheafClient,
    ServerError,
    SheafError,
    ValidationError,
)
from sheaf.modal_server import _build_asgi_app
from sheaf.spec import ModelSpec, ResourceConfig
from tests.stubs import (
    ErrorTimeSeriesBackend,
    SmokeEmbeddingBackend,
    SmokeTabularBackend,
    SmokeTimeSeriesBackend,
)


def _ts_spec(name: str = "ts") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        backend_cls=SmokeTimeSeriesBackend,
        resources=ResourceConfig(num_cpus=1),
    )


def _emb_spec(name: str = "emb") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.EMBEDDING,
        backend="_smoke_embedding",
        backend_cls=SmokeEmbeddingBackend,
        resources=ResourceConfig(num_cpus=1),
    )


def _tab_spec(name: str = "tab") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.TABULAR,
        backend="_smoke_tabular",
        backend_cls=SmokeTabularBackend,
        resources=ResourceConfig(num_cpus=1),
    )


def _err_spec(name: str = "err") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_error",
        backend_cls=ErrorTimeSeriesBackend,
        resources=ResourceConfig(num_cpus=1),
    )


def _async_client(app) -> AsyncSheafClient:  # type: ignore[no-untyped-def]
    return AsyncSheafClient(
        base_url="http://test",
        transport=httpx.ASGITransport(app=app),
    )


# ---------------------------------------------------------------------------
# Round-trip happy paths
# ---------------------------------------------------------------------------


class TestPredictRoundTrip:
    @pytest.mark.asyncio
    async def test_time_series(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        async with _async_client(app) as client:
            req = TimeSeriesRequest(
                model_name="m",
                history=[1.0, 2.0, 3.0],
                horizon=4,
                frequency=Frequency.HOURLY,
            )
            resp = await client.predict("ts", req)

        assert isinstance(resp, TimeSeriesResponse)
        assert resp.mean == [0.42] * 4
        # request_id round-trip
        assert resp.request_id == req.request_id

    @pytest.mark.asyncio
    async def test_embedding(self) -> None:
        app = _build_asgi_app([_emb_spec()])
        async with _async_client(app) as client:
            req = EmbeddingRequest(model_name="e", texts=["hello", "world"])
            resp = await client.predict("emb", req)

        assert isinstance(resp, EmbeddingResponse)
        assert resp.dim == 3
        assert len(resp.embeddings) == 2

    @pytest.mark.asyncio
    async def test_tabular(self) -> None:
        app = _build_asgi_app([_tab_spec()])
        async with _async_client(app) as client:
            req = TabularRequest(
                model_name="t",
                context_X=[[1.0, 2.0], [3.0, 4.0]],
                context_y=[0, 1],
                query_X=[[2.0, 3.0]],
                task="classification",
                output_mode="probabilities",
            )
            resp = await client.predict("tab", req)

        assert isinstance(resp, TabularResponse)
        assert resp.predictions == [0]
        assert resp.n_query == 1

    @pytest.mark.asyncio
    async def test_health_and_ready(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        async with _async_client(app) as client:
            assert await client.health("ts") == {"status": "ok"}
            ready = await client.ready("ts")
            assert ready["status"] == "ready"
            assert ready["model"] == "ts"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_wrong_model_type_raises_validation_error(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        async with _async_client(app) as client:
            wrong = TabularRequest(
                model_name="t",
                context_X=[[1.0]],
                context_y=[0],
                query_X=[[2.0]],
            )
            with pytest.raises(ValidationError) as exc_info:
                await client.predict("ts", wrong)
        assert exc_info.value.status_code == 422
        assert "time_series" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unknown_deployment_raises_sheaf_error(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        async with _async_client(app) as client:
            req = TimeSeriesRequest(
                model_name="m",
                history=[1.0],
                horizon=1,
                frequency=Frequency.HOURLY,
            )
            with pytest.raises(SheafError) as exc_info:
                await client.predict("nope", req)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_backend_exception_raises_server_error(self) -> None:
        app = _build_asgi_app([_err_spec()])
        async with _async_client(app) as client:
            req = TimeSeriesRequest(
                model_name="e",
                history=[1.0, 2.0],
                horizon=2,
                frequency=Frequency.HOURLY,
            )
            with pytest.raises(ServerError) as exc_info:
                await client.predict("err", req)
        assert exc_info.value.status_code == 500
        assert "RuntimeError" in exc_info.value.detail
        assert "backend exploded" in exc_info.value.detail
