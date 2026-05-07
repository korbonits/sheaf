"""Tests for request_id surfacing on raised SheafError exceptions.

The client mints a UUID on every BaseRequest at construction time
(``request.request_id``).  When a request fails, the raised exception
carries that UUID so callers can log-correlate without holding the
original request object — which is annoying in deeply-nested call sites
or when re-raising at a service boundary.
"""

from __future__ import annotations

import httpx
import pytest

from sheaf.api.diffusion import DiffusionRequest
from sheaf.api.time_series import Frequency, TimeSeriesRequest
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
        history=[1.0, 2.0],
        horizon=2,
        frequency=Frequency.HOURLY,
    )


class TestSyncRequestIdOnPredictErrors:
    def test_validation_error_has_request_id(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "bad"})

        with SheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ValidationError) as exc_info:
                client.predict("ts", req)

        assert exc_info.value.request_id == req.request_id

    def test_server_error_has_request_id(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "boom"})

        with SheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ServerError) as exc_info:
                client.predict("ts", req)

        assert exc_info.value.request_id == req.request_id

    def test_unknown_status_sheaf_error_has_request_id(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "no such"})

        with SheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(SheafError) as exc_info:
                client.predict("nope", req)

        assert exc_info.value.request_id == req.request_id

    def test_transport_client_error_has_request_id(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with SheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ClientError) as exc_info:
                client.predict("ts", req)

        assert exc_info.value.request_id == req.request_id

    def test_undecodable_response_client_error_has_request_id(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"bogus": "payload"})

        with SheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ClientError) as exc_info:
                client.predict("ts", req)

        assert exc_info.value.request_id == req.request_id


class TestStreamRequestId:
    @pytest.mark.asyncio
    async def test_stream_pre_stream_validation_error_has_request_id(self) -> None:
        req = DiffusionRequest(model_name="flux", prompt="x")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "bad"})

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ValidationError) as exc_info:
                async for _event in client.stream("flux", req):
                    pytest.fail("should not yield")

        assert exc_info.value.request_id == req.request_id


class TestAsyncRequestId:
    @pytest.mark.asyncio
    async def test_async_predict_validation_error_has_request_id(self) -> None:
        req = _ts_req()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "bad"})

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ValidationError) as exc_info:
                await client.predict("ts", req)

        assert exc_info.value.request_id == req.request_id


class TestSheafErrorConstructor:
    def test_request_id_default_none(self) -> None:
        e = SheafError("msg")
        assert e.request_id is None

    def test_request_id_round_trip(self) -> None:
        from uuid import uuid4

        rid = uuid4()
        e = SheafError("msg", status_code=500, request_id=rid)
        assert e.request_id == rid
        assert str(e) == "msg"
