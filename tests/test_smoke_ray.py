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

# Both fixtures below share one Ray cluster (module scope).
# Ray is initialised once here with working_dir + SHEAF_EXTRA_BACKENDS so
# both the backend_cls path and the string-registry path work correctly.


# ---------------------------------------------------------------------------
# Fixture: ModelServer lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def urls():
    """Start one Ray cluster + ModelServer with two deployments.

    smoke-ts      — backend_cls path (SmokeTimeSeriesBackend via cloudpickle)
    smoke-registry — string-registry path (_smoke_ts_registry resolved via
                     SHEAF_EXTRA_BACKENDS=tests.stubs imported at worker startup)

    Ray is initialised once with working_dir="." and SHEAF_EXTRA_BACKENDS so
    both paths are exercised on the same cluster.
    """
    import ray

    from sheaf.server import ModelServer

    ray.init(
        runtime_env={
            "working_dir": ".",
            "env_vars": {"SHEAF_EXTRA_BACKENDS": "tests.stubs"},
        }
    )

    cls_spec = ModelSpec(
        name="smoke-ts",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        backend_cls=SmokeTimeSeriesBackend,  # cloudpickled to worker
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    reg_spec = ModelSpec(
        name="smoke-registry",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts_registry",  # string lookup — no backend_cls
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )

    server = ModelServer(
        models=[cls_spec, reg_spec], host="127.0.0.1", port=_SMOKE_PORT
    )
    server.run()

    base_cls = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-ts"
    base_reg = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-registry"

    for label, url in [("smoke-ts", base_cls), ("smoke-registry", base_reg)]:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.5)
        else:
            server.shutdown()
            ray.shutdown()
            pytest.fail(f"{label} did not become ready within 30s")

    yield {"cls": base_cls, "registry": base_reg}

    server.shutdown()
    ray.shutdown()


# Convenience fixtures that unpack from the combined urls fixture
@pytest.fixture(scope="module")
def serving_url(urls: dict) -> str:
    return urls["cls"]


@pytest.fixture(scope="module")
def registry_serving_url(urls: dict) -> str:
    return urls["registry"]


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


def test_registry_path_predict(registry_serving_url: str) -> None:
    """Backend resolved via string registry (no backend_cls).

    Verifies that SHEAF_EXTRA_BACKENDS causes tests.stubs to be imported
    in the Ray worker, making '_smoke_ts_registry' available in the registry.
    A ValueError ('Unknown backend') here means the fix is broken.
    """
    payload = {
        "model_type": "time_series",
        "model_name": "smoke-registry",
        "history": [1.0, 2.0, 3.0],
        "horizon": 3,
        "frequency": "1h",
    }
    r = requests.post(f"{registry_serving_url}/predict", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["mean"] == [0.99, 0.99, 0.99]


def test_batch_policy_applied(serving_url: str) -> None:
    """BatchPolicy.max_batch_size and timeout_ms are wired into the deployment.

    Fires 16 concurrent requests against a deployment whose batch policy was
    left at the default (max_batch_size=32, timeout_ms=50).  All requests must
    succeed, which confirms set_max_batch_size / set_batch_wait_timeout_s were
    called without error during __init__.
    """
    import concurrent.futures

    payload = {
        "model_type": "time_series",
        "model_name": "smoke-ts",
        "history": [1.0, 2.0, 3.0],
        "horizon": 2,
        "frequency": "1d",
    }

    def _call() -> tuple[int, list]:
        r = requests.post(f"{serving_url}/predict", json=payload)
        return r.status_code, r.json().get("mean", [])

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_call) for _ in range(16)]
        results = [f.result() for f in futures]

    statuses = [s for s, _ in results]
    means = [m for _, m in results]
    assert all(s == 200 for s in statuses), statuses
    assert all(m == [0.42, 0.42] for m in means), means
