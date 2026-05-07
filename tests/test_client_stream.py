"""Tests for AsyncSheafClient.stream() — SSE event parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from sheaf.api.diffusion import DiffusionRequest
from sheaf.client import AsyncSheafClient, ClientError, ValidationError


def _diff_req() -> DiffusionRequest:
    return DiffusionRequest(model_name="flux", prompt="a cat")


def _sse_body(events: list[dict]) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)


class TestStreamHappy:
    @pytest.mark.asyncio
    async def test_yields_events_in_order(self) -> None:
        events_to_send = [
            {"type": "progress", "step": 1, "total_steps": 4, "done": False},
            {"type": "progress", "step": 2, "total_steps": 4, "done": False},
            {"type": "result", "done": True, "image_b64": "abc", "seed": 7},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse_body(events_to_send),
                headers={"content-type": "text/event-stream"},
            )

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            received = [event async for event in client.stream("flux", _diff_req())]

        assert received == events_to_send

    @pytest.mark.asyncio
    async def test_in_band_error_event_is_yielded(self) -> None:
        """Backend exceptions become an SSE error event, not an HTTP error."""
        events = [
            {"type": "progress", "step": 1, "total_steps": 4, "done": False},
            {"type": "error", "error": "RuntimeError: oops"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=_sse_body(events),
                headers={"content-type": "text/event-stream"},
            )

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            received = [event async for event in client.stream("flux", _diff_req())]

        assert received == events
        # Caller is responsible for checking event["type"] == "error".


class TestStreamPreStreamErrors:
    @pytest.mark.asyncio
    async def test_422_before_stream_raises(self) -> None:
        """422 from the server (before any event) → ValidationError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "wrong model_type"})

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ValidationError) as exc_info:
                async for _event in client.stream("flux", _diff_req()):
                    pytest.fail("should have raised before any event")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_500_before_stream_raises(self) -> None:
        from sheaf.client import ServerError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "boom"})

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ServerError):
                async for _event in client.stream("flux", _diff_req()):
                    pytest.fail("should have raised before any event")


class TestStreamMalformed:
    @pytest.mark.asyncio
    async def test_malformed_sse_payload_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"data: {not valid json}\n\n",
                headers={"content-type": "text/event-stream"},
            )

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(ClientError, match="Malformed SSE"):
                async for _event in client.stream("flux", _diff_req()):
                    pass

    @pytest.mark.asyncio
    async def test_non_data_lines_are_ignored(self) -> None:
        """SSE has comment lines (starting with ":") and blank lines.

        Our client only acts on ``data: `` prefixed lines.
        """
        body = (
            b": keepalive\n\n"
            b"event: tick\n\n"
            b'data: {"type": "progress", "step": 1, '
            b'"total_steps": 4, "done": false}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )

        async with AsyncSheafClient(
            base_url="http://test", transport=httpx.MockTransport(handler)
        ) as client:
            events = [event async for event in client.stream("flux", _diff_req())]

        assert len(events) == 1
        assert events[0]["step"] == 1
