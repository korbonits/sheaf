"""Tests for RetryConfig + retry/backoff logic on SheafClient + AsyncSheafClient.

Counts handler invocations to verify retry behavior; sleep is mocked so the
tests run fast and deterministic backoff math can be asserted.
"""

from __future__ import annotations

import httpx
import pytest

from sheaf.api.time_series import Frequency, TimeSeriesRequest
from sheaf.client import (
    AsyncSheafClient,
    ClientError,
    RetryConfig,
    ServerError,
    SheafClient,
)


def _ts_req() -> TimeSeriesRequest:
    return TimeSeriesRequest(
        model_name="m",
        history=[1.0, 2.0, 3.0],
        horizon=3,
        frequency=Frequency.HOURLY,
    )


def _ok_body(req: TimeSeriesRequest) -> dict:
    return {
        "request_id": str(req.request_id),
        "model_type": "time_series",
        "model_name": req.model_name,
        "horizon": req.horizon,
        "frequency": req.frequency.value,
        "mean": [0.5] * req.horizon,
    }


# ---------------------------------------------------------------------------
# RetryConfig validation
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_defaults(self) -> None:
        rc = RetryConfig()
        assert rc.max_attempts == 1
        assert rc.backoff_factor == 0.5
        assert rc.retry_on_status == (502, 503, 504)
        assert rc.retry_on_connection_errors is True

    def test_zero_max_attempts_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            RetryConfig(max_attempts=0)

    def test_negative_backoff_rejected(self) -> None:
        with pytest.raises(ValueError, match="backoff_factor"):
            RetryConfig(backoff_factor=-1)

    def test_sleep_seconds_first_attempt_zero(self) -> None:
        rc = RetryConfig(backoff_factor=0.5)
        assert rc.sleep_seconds(0) == 0.0

    def test_sleep_seconds_exponential(self) -> None:
        rc = RetryConfig(backoff_factor=0.5)
        # Before attempt 1 (after the 1st failed): 0.5
        # Before attempt 2 (after the 2nd failed): 1.0
        # Before attempt 3: 2.0
        assert rc.sleep_seconds(1) == 0.5
        assert rc.sleep_seconds(2) == 1.0
        assert rc.sleep_seconds(3) == 2.0


# ---------------------------------------------------------------------------
# Sync retry
# ---------------------------------------------------------------------------


class TestSyncRetry:
    def test_no_retry_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without RetryConfig, a 503 fires once and raises ServerError."""
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={"detail": "transient"})

        with SheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ServerError):
                client.predict("ts", _ts_req())

        assert calls["n"] == 1
        assert slept == []

    def test_retry_succeeds_on_second_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        req = _ts_req()
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"detail": "transient"})
            return httpx.Response(200, json=_ok_body(req))

        retry = RetryConfig(max_attempts=3, backoff_factor=0.5)
        with SheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            resp = client.predict("ts", req)

        assert calls["n"] == 2
        # One sleep before attempt #2: backoff_factor * 2^(1-1) = 0.5
        assert slept == [0.5]
        assert resp.mean == [0.5] * 3

    def test_retry_exhausts_then_returns_last_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={"detail": "still down"})

        retry = RetryConfig(max_attempts=3, backoff_factor=0.5)
        with SheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(ServerError) as exc_info:
                client.predict("ts", _ts_req())

        assert calls["n"] == 3
        # Two sleeps before attempts 2 and 3: 0.5, 1.0
        assert slept == [0.5, 1.0]
        assert exc_info.value.status_code == 503

    def test_4xx_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """422 is a client error — retrying won't help."""
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(422, json={"detail": "bad request"})

        retry = RetryConfig(max_attempts=3)
        with SheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            from sheaf.client import ValidationError

            with pytest.raises(ValidationError):
                client.predict("ts", _ts_req())

        assert calls["n"] == 1
        assert slept == []

    def test_connection_error_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))

        req = _ts_req()
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json=_ok_body(req))

        retry = RetryConfig(max_attempts=3, backoff_factor=0.1)
        with SheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            resp = client.predict("ts", req)

        assert calls["n"] == 2
        assert slept == [0.1]
        assert resp.mean == [0.5] * 3

    def test_connection_error_not_retried_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("time.sleep", lambda s: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("refused")

        retry = RetryConfig(max_attempts=3, retry_on_connection_errors=False)
        with SheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(ClientError):
                client.predict("ts", _ts_req())

        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Async retry
# ---------------------------------------------------------------------------


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_async_retry_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slept: list[float] = []

        async def fake_sleep(s: float) -> None:
            slept.append(s)

        monkeypatch.setattr("asyncio.sleep", fake_sleep)

        req = _ts_req()
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(502, json={"detail": "down"})
            return httpx.Response(200, json=_ok_body(req))

        retry = RetryConfig(max_attempts=4, backoff_factor=0.5)
        async with AsyncSheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            resp = await client.predict("ts", req)

        assert calls["n"] == 3
        assert slept == [0.5, 1.0]
        assert resp.mean == [0.5] * 3


# ---------------------------------------------------------------------------
# Health/ready also retry
# ---------------------------------------------------------------------------


class TestHealthReadyRetry:
    def test_health_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("time.sleep", lambda s: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"detail": "warming up"})
            return httpx.Response(200, json={"status": "ok"})

        retry = RetryConfig(max_attempts=2)
        with SheafClient(
            base_url="http://test",
            retry=retry,
            transport=httpx.MockTransport(handler),
        ) as client:
            assert client.health("ts") == {"status": "ok"}
        assert calls["n"] == 2
