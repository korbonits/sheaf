"""End-to-end smoke test for the Ray Serve integration.

Spins up a real Ray cluster + Serve instance using a stub backend
(no model weights required), sends HTTP requests, then tears down.

Run explicitly:
    uv run pytest tests/test_smoke_ray.py -v -s

Skipped in normal CI because it starts a local Ray cluster (~5s overhead).
Set SHEAF_SMOKE_TEST=1 to enable, or pass --smoke to pytest.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

from sheaf.api.base import ModelType
from sheaf.spec import ModelSpec, ResourceConfig
from tests.stubs import SmokeTimeSeriesBackend

# ---------------------------------------------------------------------------
# Opt-in gate
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("SHEAF_SMOKE_TEST"),
    reason="Set SHEAF_SMOKE_TEST=1 to run Ray Serve smoke tests",
)

_SMOKE_PORT = 8787  # avoid clashing with anything on 8000


# ---------------------------------------------------------------------------
# Fixture: ModelServer lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def serving_url():
    """Start ModelServer, yield base URL, shut down after all tests."""
    import ray

    from sheaf.server import ModelServer

    spec = ModelSpec(
        name="smoke-ts",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        backend_cls=SmokeTimeSeriesBackend,  # cloudpickled to worker
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )

    server = ModelServer(models=[spec], host="127.0.0.1", port=_SMOKE_PORT)
    server.run()

    base = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-ts"

    # Wait for Serve to be ready (up to 30s)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        server.shutdown()
        ray.shutdown()
        pytest.fail("Ray Serve did not become ready within 30s")

    yield base

    server.shutdown()
    ray.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(serving_url: str) -> None:
    r = requests.get(f"{serving_url}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready(serving_url: str) -> None:
    r = requests.get(f"{serving_url}/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model"] == "smoke-ts"


def test_predict_mean(serving_url: str) -> None:
    payload = {
        "model_type": "time_series",
        "model_name": "smoke-ts",
        "history": [1.0, 2.0, 3.0, 4.0, 5.0],
        "horizon": 6,
        "frequency": "1h",
    }
    r = requests.post(f"{serving_url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mean"] == [0.42] * 6
    assert body["horizon"] == 6
    assert "request_id" in body


def test_predict_wrong_model_type_returns_422(serving_url: str) -> None:
    payload = {
        "model_type": "tabular",
        "model_name": "smoke-ts",
        "context_X": [[1.0, 2.0]],
        "context_y": [0],
        "query_X": [[3.0, 4.0]],
    }
    r = requests.post(f"{serving_url}/predict", json=payload)
    assert r.status_code == 422


def test_predict_malformed_body_returns_422(serving_url: str) -> None:
    r = requests.post(f"{serving_url}/predict", json={"model_type": "time_series"})
    assert r.status_code == 422


def test_predict_concurrent(serving_url: str) -> None:
    """Fire 8 requests concurrently — exercises the @serve.batch path."""
    import concurrent.futures

    payload = {
        "model_type": "time_series",
        "model_name": "smoke-ts",
        "history": [1.0, 2.0, 3.0],
        "horizon": 4,
        "frequency": "1d",
    }

    def _call() -> int:
        return requests.post(f"{serving_url}/predict", json=payload).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_call) for _ in range(8)]
        statuses = [f.result() for f in futures]

    assert all(s == 200 for s in statuses), statuses
