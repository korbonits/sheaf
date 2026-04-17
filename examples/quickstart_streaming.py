"""Streaming responses quickstart.

Demonstrates Sheaf's SSE streaming endpoint.  Uses lightweight stub backends
so no model weights are required — run it immediately after ``pip install sheaf-serve``.

Key points:
  - ``POST /{name}/stream`` returns a ``text/event-stream`` response.
  - Each event is a ``data: <json>\\n\\n`` line.
  - Two event shapes:
      - Progress: ``{"type": "progress", "step": N, "total_steps": N, "done": false}``
      - Result:   ``{"type": "result",   "done": true,  ...response_fields}``
  - Backends that don't override ``stream_predict`` yield a single result event —
    identical to a regular ``/predict`` call but as SSE.
  - Backends like FLUX override ``stream_predict`` to emit per-step progress events
    before the final image.

Usage::

    python examples/quickstart_streaming.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.testclient import TestClient

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.modal_server import _build_asgi_app
from sheaf.registry import register_backend
from sheaf.spec import ModelSpec

# ---------------------------------------------------------------------------
# 1. Simple backend (uses default stream_predict — single result event)
# ---------------------------------------------------------------------------


@register_backend("_demo_stream_simple")
class _SimpleTSBackend(ModelBackend):
    """Returns the running mean of history as a flat forecast."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, TimeSeriesRequest)
        mean_val = sum(request.history) / len(request.history)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[round(mean_val, 4)] * request.horizon,
        )


# ---------------------------------------------------------------------------
# 2. Progressive backend (overrides stream_predict — emits progress events)
# ---------------------------------------------------------------------------


@register_backend("_demo_stream_progressive")
class _ProgressiveTSBackend(ModelBackend):
    """Simulates a multi-step inference process with progress events."""

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
            mean=[42.0] * request.horizon,
        )

    async def stream_predict(self, request: BaseRequest) -> Any:  # type: ignore[override]
        # Emit N progress events before the final result.
        n_steps = 5
        for step in range(1, n_steps + 1):
            await asyncio.sleep(0.01)  # simulate compute
            yield {
                "type": "progress",
                "step": step,
                "total_steps": n_steps,
                "done": False,
            }
        result = self.predict(request)
        yield {"type": "result", "done": True, **result.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# 3. Parse SSE helper
# ---------------------------------------------------------------------------


def parse_sse(body: str) -> list[dict[str, Any]]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# ---------------------------------------------------------------------------
# 4. Build the ASGI app
# ---------------------------------------------------------------------------

specs = [
    ModelSpec(
        name="simple",
        model_type=ModelType.TIME_SERIES,
        backend="_demo_stream_simple",
    ),
    ModelSpec(
        name="progressive",
        model_type=ModelType.TIME_SERIES,
        backend="_demo_stream_progressive",
    ),
]
app = _build_asgi_app(specs)
client = TestClient(app)

payload = {
    "model_type": "time_series",
    "model_name": "demo",
    "history": [100.0, 110.0, 120.0, 130.0, 140.0],
    "horizon": 4,
    "frequency": "1d",
}

# ---------------------------------------------------------------------------
# 5. Default streaming (single result event)
# ---------------------------------------------------------------------------

print("=== Simple backend (default stream_predict) ===")
with client.stream("POST", "/simple/stream", json=payload) as resp:
    resp.read()
    events = parse_sse(resp.text)

for event in events:
    print(f"  {event}")

assert len(events) == 1
assert events[0]["type"] == "result"
assert events[0]["done"] is True
print(f"  → {len(events)} event  (mean: {events[0]['mean']})")

# ---------------------------------------------------------------------------
# 6. Progressive streaming (progress events + result)
# ---------------------------------------------------------------------------

print("\n=== Progressive backend (overridden stream_predict) ===")
with client.stream("POST", "/progressive/stream", json=payload) as resp:
    resp.read()
    events = parse_sse(resp.text)

for event in events:
    if event["type"] == "progress":
        bar = "█" * event["step"] + "░" * (event["total_steps"] - event["step"])
        print(f"  [{bar}] step {event['step']}/{event['total_steps']}")
    else:
        print(f"  result: mean={event['mean']}, done={event['done']}")

progress_events = [e for e in events if e["type"] == "progress"]
result_events = [e for e in events if e["type"] == "result"]
print(f"  → {len(progress_events)} progress events + {len(result_events)} result event")

# ---------------------------------------------------------------------------
# 7. Compare /predict and /stream (same result, different delivery)
# ---------------------------------------------------------------------------

print("\n=== /predict vs /stream — same result ===")
predict_resp = client.post("/simple/predict", json=payload)
predict_mean = predict_resp.json()["mean"]

with client.stream("POST", "/simple/stream", json=payload) as resp:
    resp.read()
    stream_result = next(e for e in parse_sse(resp.text) if e["type"] == "result")
stream_mean = stream_result["mean"]

assert predict_mean == stream_mean
print(f"  /predict mean:  {predict_mean}")
print(f"  /stream  mean:  {stream_mean}")
print("  ✓ identical")

print("\nDone.")
