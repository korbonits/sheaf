"""Request caching quickstart.

Demonstrates Sheaf's in-process LRU response cache.  Uses a lightweight
stub backend so no model weights are required — run it immediately after
``pip install sheaf-serve``.

Key points:
  - Add ``cache=CacheConfig(enabled=True)`` to any ``ModelSpec``.
  - The first call runs inference; identical subsequent calls are served
    from the in-process cache.
  - ``ttl_s`` expires entries after N seconds.
  - ``exclude_fields`` lets callers omit fields from the key (e.g. ``seed``
    for diffusion models, so same-seed repeats hit but different seeds miss).
  - ``SHEAF_CACHE_DISABLED=1`` env var disables all caches globally (useful
    in integration test runs where you need fresh backend output every time).

Good candidates for caching:
  - Embedding models: same image / text → same vector.
  - Time-series forecasts with fixed history.
  - Any model where the input fully determines the output.

Poor candidates:
  - Diffusion with random seeds *unless* you include the seed in the key
    (omit ``"seed"`` from ``exclude_fields`` and the same seed is cached).
  - Models where the caller explicitly needs fresh output each time.

Usage::

    python examples/quickstart_cache.py
"""

from __future__ import annotations

import time

from starlette.testclient import TestClient

from sheaf.api.base import ModelType

# ---------------------------------------------------------------------------
# Register a trivial stub backend (no weights needed)
# ---------------------------------------------------------------------------
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse  # noqa: E402
from sheaf.backends.base import ModelBackend  # noqa: E402
from sheaf.cache import CacheConfig
from sheaf.modal_server import _build_asgi_app
from sheaf.registry import register_backend  # noqa: E402
from sheaf.spec import ModelSpec


@register_backend("_demo_ts")
class _DemoTSBackend(ModelBackend):
    """Minimal time-series backend: returns 0.0 for every horizon step."""

    def load(self) -> None:
        # Simulate a small load delay so the first predict() is visibly slower.
        time.sleep(0.05)

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: TimeSeriesRequest) -> TimeSeriesResponse:  # type: ignore[override]
        # Simulate inference latency.
        time.sleep(0.1)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[0.0] * request.horizon,
        )


# ---------------------------------------------------------------------------
# 1. ModelSpec with caching enabled
# ---------------------------------------------------------------------------

spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="_demo_ts",
    cache=CacheConfig(
        enabled=True,
        max_size=512,  # keep up to 512 distinct request fingerprints
        ttl_s=300.0,  # expire entries after 5 minutes
    ),
)

# ---------------------------------------------------------------------------
# 2. Build the ASGI app (works without Ray — same HTTP contract as ModelServer)
# ---------------------------------------------------------------------------

print("Building ASGI app and loading backend…")
app = _build_asgi_app([spec])
client = TestClient(app)

# ---------------------------------------------------------------------------
# 3. Define a request payload
# ---------------------------------------------------------------------------

payload = {
    "model_type": "time_series",
    "model_name": "forecaster",
    "history": [312, 298, 275, 260, 255, 263, 285, 320, 368, 402, 421, 435],
    "horizon": 6,
    "frequency": "1h",
}

# ---------------------------------------------------------------------------
# 4. First call — backend runs (cache miss)
# ---------------------------------------------------------------------------

print("\nCall 1  (cache miss — backend executes):")
t0 = time.perf_counter()
r = client.post("/forecaster/predict", json=payload)
r.raise_for_status()
latency_first = (time.perf_counter() - t0) * 1000
print(f"  mean: {r.json()['mean']}")
print(f"  latency: {latency_first:.1f} ms")

# ---------------------------------------------------------------------------
# 5. Second call — same payload, different request_id → cache hit
# ---------------------------------------------------------------------------

print("\nCall 2  (cache hit — backend skipped):")
t0 = time.perf_counter()
r = client.post("/forecaster/predict", json=payload)
r.raise_for_status()
latency_cached = (time.perf_counter() - t0) * 1000
print(f"  mean: {r.json()['mean']}")
print(f"  latency: {latency_cached:.1f} ms")

speedup = latency_first / max(latency_cached, 0.01)
print(
    f"\n  Speedup: {speedup:.1f}×  ({latency_first:.1f} ms → {latency_cached:.1f} ms)"
)  # noqa: E501

# ---------------------------------------------------------------------------
# 6. Different payload → cache miss (distinct entry)
# ---------------------------------------------------------------------------

different_payload = {**payload, "history": [100, 110, 120, 130, 140, 150]}

print("\nCall 3  (different history — cache miss, new entry):")
t0 = time.perf_counter()
r = client.post("/forecaster/predict", json=different_payload)
r.raise_for_status()
print(f"  mean: {r.json()['mean']}")
print(f"  latency: {(time.perf_counter() - t0) * 1000:.1f} ms")

# ---------------------------------------------------------------------------
# 7. exclude_fields — omit a field from the cache key
# ---------------------------------------------------------------------------

print("\n--- exclude_fields demo ---")
print("Two requests that differ only in 'model_name' → treated as the same key.")

spec_excl = ModelSpec(
    name="forecaster-excl",
    model_type=ModelType.TIME_SERIES,
    backend="_demo_ts",
    cache=CacheConfig(
        enabled=True,
        exclude_fields=["model_name"],  # model_name excluded from key
    ),
)
app_excl = _build_asgi_app([spec_excl])
client_excl = TestClient(app_excl)

r1 = client_excl.post(
    "/forecaster-excl/predict", json={**payload, "model_name": "version-a"}
)
r2 = client_excl.post(
    "/forecaster-excl/predict", json={**payload, "model_name": "version-b"}
)
r1.raise_for_status()
r2.raise_for_status()
print(f"  version-a mean: {r1.json()['mean']}")
print(f"  version-b mean: {r2.json()['mean']}  (served from cache — same result)")

print("\nDone.")
