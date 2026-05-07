"""Unit tests for SheafClient / AsyncSheafClient using httpx.MockTransport.

No real HTTP server, no FastAPI app — just a recorded request/response.
The integration test in ``test_client_integration.py`` exercises the same
client against the real ``_build_asgi_app`` to confirm the ASGI wire-up.
"""

from __future__ import annotations

import json

import httpx
import pytest

from sheaf.api.time_series import Frequency, TimeSeriesRequest, TimeSeriesResponse
from sheaf.client import (
    AsyncSheafClient,
    ClientError,
    ServerError,
    SheafClient,
    SheafError,
    ValidationError,
)


def _ts_req() -> TimeSeriesRequest:
    return TimeSeriesRequest(
        model_name="m",
        history=[1.0, 2.0, 3.0],
        horizon=3,
        frequency=Frequency.HOURLY,
    )


def _ts_response_body(req: TimeSeriesRequest) -> dict:
    return {
        "request_id": str(req.request_id),
        "model_type": "time_series",
        "model_name": req.model_name,
        "horizon": req.horizon,
        "frequency": req.frequency.value,
        "mean": [0.5, 0.5, 0.5],
    }


def _make_transport(
    handler,  # type: ignore[no-untyped-def]
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class TestSheafClientPredict:
    def test_predict_decodes_typed_response(self) -> None:
        req = _ts_req()
        body = _ts_response_body(req)

        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=body)

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            resp = client.predict("ts", req)

        assert isinstance(resp, TimeSeriesResponse)
        assert resp.mean == [0.5, 0.5, 0.5]
        assert resp.horizon == 3
        # Check the wire format matches what the server expects
        assert captured["method"] == "POST"
        assert captured["url"] == "http://test/ts/predict"
        assert captured["body"]["model_type"] == "time_series"
        assert captured["body"]["history"] == [1.0, 2.0, 3.0]

    def test_predict_422_raises_validation_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422, json={"detail": "Backend 'x' expects time_series, got tabular"}
            )

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(ValidationError) as exc_info:
                client.predict("x", _ts_req())

        assert exc_info.value.status_code == 422
        assert "time_series" in exc_info.value.detail

    def test_predict_500_raises_server_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500, json={"detail": "RuntimeError: backend exploded"}
            )

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(ServerError) as exc_info:
                client.predict("x", _ts_req())

        assert exc_info.value.status_code == 500
        assert "backend exploded" in exc_info.value.detail

    def test_predict_unexpected_status_raises_sheaf_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "No deployment named 'x'"})

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(SheafError) as exc_info:
                client.predict("x", _ts_req())

        # Not a ValidationError or ServerError — base SheafError
        assert not isinstance(exc_info.value, (ValidationError, ServerError))
        assert exc_info.value.status_code == 404

    def test_predict_non_json_200_raises_client_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(ClientError):
                client.predict("x", _ts_req())

    def test_predict_undecodable_response_raises_client_error(self) -> None:
        """Server returned 200 + JSON that doesn't match any AnyResponse member."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"bogus": "payload"})

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(ClientError):
                client.predict("x", _ts_req())

    def test_transport_error_raises_client_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(ClientError, match="ConnectError"):
                client.predict("x", _ts_req())


class TestSheafClientHealthReady:
    def test_health_returns_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url).endswith("/ts/health")
            return httpx.Response(200, json={"status": "ok"})

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            assert client.health("ts") == {"status": "ok"}

    def test_ready_returns_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url).endswith("/ts/ready")
            return httpx.Response(200, json={"status": "ready", "model": "ts"})

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            body = client.ready("ts")
            assert body["status"] == "ready"
            assert body["model"] == "ts"

    def test_health_404_raises_sheaf_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "no such deployment"})

        with SheafClient(
            base_url="http://test", transport=_make_transport(handler)
        ) as client:
            with pytest.raises(SheafError) as exc_info:
                client.health("nope")
        assert exc_info.value.status_code == 404


class TestSheafClientLifecycle:
    def test_close_idempotent(self) -> None:
        client = SheafClient(base_url="http://test")
        client.close()
        client.close()  # second close shouldn't raise

    def test_context_manager_closes(self) -> None:
        with SheafClient(base_url="http://test") as c:
            assert c._http is not None  # type: ignore[has-type]
        # After exit, client.close() was called — httpx clients are reusable
        # post-close in the sense that they don't error on close-twice.

    def test_custom_headers_passed_through(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization", "")
            return httpx.Response(200, json=_ts_response_body(_ts_req()))

        with SheafClient(
            base_url="http://test",
            headers={"authorization": "Bearer abc"},
            transport=_make_transport(handler),
        ) as client:
            client.predict("ts", _ts_req())

        assert captured["auth"] == "Bearer abc"


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class TestAsyncSheafClient:
    @pytest.mark.asyncio
    async def test_predict(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ts_response_body(req))

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            resp = await client.predict("ts", req)

        assert isinstance(resp, TimeSeriesResponse)
        assert resp.mean == [0.5, 0.5, 0.5]

    @pytest.mark.asyncio
    async def test_predict_422(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "bad"})

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ValidationError):
                await client.predict("x", _ts_req())

    @pytest.mark.asyncio
    async def test_health(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok"})

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            assert await client.health("ts") == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timeout")

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ClientError, match="ReadTimeout"):
                await client.predict("x", _ts_req())
