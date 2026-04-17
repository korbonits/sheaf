"""Tests for streaming responses — base default, FluxBackend override, SSE endpoint."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import sys
import zlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.diffusion import DiffusionRequest
from sheaf.api.time_series import Frequency, TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(width: int = 8, height: int = 8) -> bytes:
    """Minimal valid RGB PNG — no PIL required."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    raw = b"\x00" + b"\xff\xff\xff" * width
    compressed = zlib.compress(raw * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def _ts_req(horizon: int = 3) -> TimeSeriesRequest:
    return TimeSeriesRequest(
        model_name="m",
        history=[1.0, 2.0, 3.0],
        horizon=horizon,
        frequency=Frequency.DAILY,
    )


# ---------------------------------------------------------------------------
# Stub backend: streams 3 progress events then 1 result
# ---------------------------------------------------------------------------


@register_backend("_test_streaming_ts")
class _StreamingTSBackend(ModelBackend):
    """Time-series backend that yields 3 progress events before the result."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        from sheaf.api.base import ModelType

        return ModelType.TIME_SERIES

    def predict(self, request: Any) -> Any:
        assert isinstance(request, TimeSeriesRequest)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[7.0] * request.horizon,
        )

    async def stream_predict(self, request: Any) -> Any:  # type: ignore[override]
        for step in range(1, 4):
            yield {"type": "progress", "step": step, "total_steps": 3, "done": False}
        result = self.predict(request)
        yield {"type": "result", "done": True, **result.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Stub backend: always raises in stream_predict
# ---------------------------------------------------------------------------


@register_backend("_test_streaming_error")
class _ErrorStreamingBackend(ModelBackend):
    """Backend that raises mid-stream."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        from sheaf.api.base import ModelType

        return ModelType.TIME_SERIES

    def predict(self, request: Any) -> Any:
        raise RuntimeError("bang")

    async def stream_predict(self, request: Any) -> Any:  # type: ignore[override]
        yield {"type": "progress", "step": 1, "total_steps": 2, "done": False}
        raise RuntimeError("stream exploded")


# ---------------------------------------------------------------------------
# Helpers for FluxBackend mocks (same pattern as test_flux_backend.py)
# ---------------------------------------------------------------------------


def _make_streaming_pipeline(
    num_steps: int = 4,
    width: int = 8,
    height: int = 8,
) -> MagicMock:
    """Pipeline mock that fires ``callback_on_step_end`` for each step."""
    fake_img = MagicMock()
    fake_img.width = width
    fake_img.height = height

    def _save(buf: io.BytesIO, format: str = "PNG") -> None:
        buf.write(_make_png(width, height))

    fake_img.save.side_effect = _save

    def _call(**kwargs: Any) -> MagicMock:
        callback = kwargs.get("callback_on_step_end")
        n = kwargs.get("num_inference_steps", num_steps)
        if callback is not None:
            for step in range(n):
                callback(None, step, None, {})
        result = MagicMock()
        result.images = [fake_img]
        return result

    instance = MagicMock()
    instance.side_effect = _call

    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value.to.return_value = instance
    return pipeline_cls


def _make_fake_torch() -> MagicMock:
    torch = MagicMock()
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.Generator.return_value.manual_seed.return_value = MagicMock()
    return torch


def _loaded_flux_backend(num_steps: int = 4) -> tuple:
    from sheaf.backends.flux import FluxBackend

    fake_pipeline = _make_streaming_pipeline(num_steps=num_steps)
    fake_diffusers = MagicMock()
    fake_diffusers.FluxPipeline = fake_pipeline
    fake_torch = _make_fake_torch()

    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        backend = FluxBackend(model_id="test", device="cpu")
        backend.load()

    backend._fake_torch = fake_torch
    return backend, fake_torch


# ---------------------------------------------------------------------------
# Default stream_predict in ModelBackend
# ---------------------------------------------------------------------------


class TestDefaultStreamPredict:
    """ModelBackend.stream_predict yields exactly one result event."""

    def _run_async(self, coro: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _collect(self, backend: ModelBackend, req: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for event in backend.stream_predict(req):
            events.append(event)
        return events

    @pytest.mark.asyncio
    async def test_default_yields_one_result_event(self) -> None:
        from tests.stubs import SmokeTimeSeriesBackend

        backend = SmokeTimeSeriesBackend()
        backend.load()
        req = _ts_req(horizon=3)
        events = await self._collect(backend, req)
        assert len(events) == 1
        assert events[0]["type"] == "result"
        assert events[0]["done"] is True

    @pytest.mark.asyncio
    async def test_default_result_contains_mean(self) -> None:
        from tests.stubs import SmokeTimeSeriesBackend

        backend = SmokeTimeSeriesBackend()
        backend.load()
        req = _ts_req(horizon=5)
        events = await self._collect(backend, req)
        assert events[0]["mean"] == [0.42] * 5

    @pytest.mark.asyncio
    async def test_default_result_has_request_id(self) -> None:
        from tests.stubs import SmokeTimeSeriesBackend

        backend = SmokeTimeSeriesBackend()
        backend.load()
        req = _ts_req()
        events = await self._collect(backend, req)
        assert events[0]["request_id"] == str(req.request_id)

    @pytest.mark.asyncio
    async def test_custom_backend_yields_progress_and_result(self) -> None:
        backend = _StreamingTSBackend()
        backend.load()
        req = _ts_req(horizon=2)
        events = await self._collect(backend, req)
        progress = [e for e in events if e["type"] == "progress"]
        result = [e for e in events if e["type"] == "result"]
        assert len(progress) == 3
        assert len(result) == 1
        assert result[0]["done"] is True


# ---------------------------------------------------------------------------
# FluxBackend.stream_predict
# ---------------------------------------------------------------------------


class TestFluxStreamPredict:
    async def _collect(self, backend: Any, req: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for event in backend.stream_predict(req):
            events.append(event)
        return events

    def _make_req(self, num_steps: int = 4) -> DiffusionRequest:
        return DiffusionRequest(
            model_name="flux",
            prompt="a red apple",
            num_inference_steps=num_steps,
            seed=42,
        )

    @pytest.mark.asyncio
    async def test_yields_progress_then_result(self) -> None:
        backend, fake_torch = _loaded_flux_backend(num_steps=4)
        req = self._make_req(num_steps=4)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        progress = [e for e in events if e["type"] == "progress"]
        result = [e for e in events if e["type"] == "result"]
        assert len(progress) == 4
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_progress_step_numbers_are_sequential(self) -> None:
        backend, fake_torch = _loaded_flux_backend(num_steps=3)
        req = self._make_req(num_steps=3)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        steps = [e["step"] for e in events if e["type"] == "progress"]
        assert steps == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_progress_total_steps_matches_request(self) -> None:
        backend, fake_torch = _loaded_flux_backend(num_steps=6)
        req = self._make_req(num_steps=6)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        totals = {e["total_steps"] for e in events if e["type"] == "progress"}
        assert totals == {6}

    @pytest.mark.asyncio
    async def test_progress_done_is_false(self) -> None:
        backend, fake_torch = _loaded_flux_backend(num_steps=2)
        req = self._make_req(num_steps=2)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        for e in events:
            if e["type"] == "progress":
                assert e["done"] is False

    @pytest.mark.asyncio
    async def test_result_done_is_true(self) -> None:
        backend, fake_torch = _loaded_flux_backend()
        req = self._make_req()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        result = next(e for e in events if e["type"] == "result")
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_result_contains_image_b64(self) -> None:
        backend, fake_torch = _loaded_flux_backend()
        req = self._make_req()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        result = next(e for e in events if e["type"] == "result")
        raw = base64.b64decode(result["image_b64"])
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.asyncio
    async def test_result_seed_matches_request(self) -> None:
        backend, fake_torch = _loaded_flux_backend()
        req = self._make_req()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        result = next(e for e in events if e["type"] == "result")
        assert result["seed"] == 42

    @pytest.mark.asyncio
    async def test_result_is_last_event(self) -> None:
        backend, fake_torch = _loaded_flux_backend(num_steps=2)
        req = self._make_req(num_steps=2)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        assert events[-1]["type"] == "result"

    @pytest.mark.asyncio
    async def test_wrong_request_type_raises(self) -> None:
        backend, fake_torch = _loaded_flux_backend()
        req = _ts_req()
        with pytest.raises(TypeError, match="DiffusionRequest"):
            async for _ in backend.stream_predict(req):
                pass

    @pytest.mark.asyncio
    async def test_one_step_yields_one_progress_event(self) -> None:
        backend, fake_torch = _loaded_flux_backend(num_steps=1)
        req = self._make_req(num_steps=1)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            events = await self._collect(backend, req)
        progress = [e for e in events if e["type"] == "progress"]
        assert len(progress) == 1
        assert progress[0]["step"] == 1


# ---------------------------------------------------------------------------
# SSE endpoint — /stream via _build_asgi_app + TestClient
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse SSE body into a list of event dicts."""
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class TestStreamEndpoint:
    """HTTP-level tests for POST /{name}/stream via the modal ASGI builder."""

    def _make_client(self, backend_name: str, model_type_str: str) -> Any:
        from starlette.testclient import TestClient

        from sheaf.modal_server import _build_asgi_app
        from sheaf.spec import ModelSpec

        spec = ModelSpec(
            name="test-model",
            model_type=model_type_str,
            backend=backend_name,
        )
        app = _build_asgi_app([spec])
        return TestClient(app)

    def test_stream_returns_200_with_sse_content_type(self) -> None:
        client = self._make_client("_test_streaming_ts", "time_series")
        payload = {
            "model_type": "time_series",
            "model_name": "m",
            "history": [1.0, 2.0, 3.0],
            "horizon": 3,
            "frequency": "1d",
        }
        with client.stream("POST", "/test-model/stream", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_yields_progress_and_result_events(self) -> None:
        client = self._make_client("_test_streaming_ts", "time_series")
        payload = {
            "model_type": "time_series",
            "model_name": "m",
            "history": [1.0, 2.0, 3.0],
            "horizon": 2,
            "frequency": "1d",
        }
        with client.stream("POST", "/test-model/stream", json=payload) as resp:
            resp.read()
            body = resp.text

        events = _parse_sse(body)
        progress = [e for e in events if e["type"] == "progress"]
        result = [e for e in events if e["type"] == "result"]
        assert len(progress) == 3
        assert len(result) == 1

    def test_stream_result_done_is_true(self) -> None:
        client = self._make_client("_test_streaming_ts", "time_series")
        payload = {
            "model_type": "time_series",
            "model_name": "m",
            "history": [1.0, 2.0, 3.0],
            "horizon": 3,
            "frequency": "1d",
        }
        with client.stream("POST", "/test-model/stream", json=payload) as resp:
            resp.read()
            body = resp.text

        events = _parse_sse(body)
        result = next(e for e in events if e["type"] == "result")
        assert result["done"] is True
        assert result["mean"] == [7.0, 7.0, 7.0]

    def test_stream_404_for_unknown_deployment(self) -> None:
        client = self._make_client("_test_streaming_ts", "time_series")
        payload = {
            "model_type": "time_series",
            "model_name": "m",
            "history": [1.0],
            "horizon": 1,
            "frequency": "1d",
        }
        resp = client.post("/no-such-model/stream", json=payload)
        assert resp.status_code == 404

    def test_stream_422_for_wrong_model_type(self) -> None:
        client = self._make_client("_test_streaming_ts", "time_series")
        # Send a tabular request to a time-series deployment
        payload = {
            "model_type": "tabular",
            "model_name": "m",
            "task": "classification",
            "context_X": [[1, 2]],
            "context_y": [0],
            "query_X": [[1, 2]],
        }
        resp = client.post("/test-model/stream", json=payload)
        assert resp.status_code == 422

    def test_stream_error_yields_error_event(self) -> None:
        client = self._make_client("_test_streaming_error", "time_series")
        payload = {
            "model_type": "time_series",
            "model_name": "m",
            "history": [1.0, 2.0, 3.0],
            "horizon": 3,
            "frequency": "1d",
        }
        with client.stream("POST", "/test-model/stream", json=payload) as resp:
            resp.read()
            body = resp.text

        events = _parse_sse(body)
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "stream exploded" in error_events[0]["error"]

    def test_stream_default_backend_yields_single_result(self) -> None:
        """Default stream_predict (not overridden) yields one result event."""
        client = self._make_client("_smoke_ts", "time_series")
        payload = {
            "model_type": "time_series",
            "model_name": "m",
            "history": [1.0, 2.0, 3.0],
            "horizon": 3,
            "frequency": "1d",
        }
        with client.stream("POST", "/test-model/stream", json=payload) as resp:
            resp.read()
            body = resp.text

        events = _parse_sse(body)
        assert len(events) == 1
        assert events[0]["type"] == "result"
        assert events[0]["done"] is True
