"""End-to-end smoke test for the Ray Serve integration.

Spins up a real Ray cluster + Serve instance using a stub backend
(no model weights required), sends HTTP requests, then tears down.

Run explicitly:
    uv run pytest tests/test_smoke_ray.py -v -s

Skipped in normal CI because it starts a local Ray cluster (~5s overhead).
Set SHEAF_SMOKE_TEST=1 to enable, or pass --smoke to pytest.
"""

from __future__ import annotations

import base64
import os
import time

import pytest
import requests

from sheaf.api.base import ModelType
from sheaf.spec import ModelSpec, ResourceConfig
from tests.stubs import (
    ErrorTimeSeriesBackend,
    SmokeDepthBackend,
    SmokeDetectionBackend,
    SmokeEmbeddingBackend,
    SmokeMolecularBackend,
    SmokeSegmentationBackend,
    SmokeTimeSeriesBackend,
    SmokeWeatherBackend,
)

_FAKE_IMG_B64 = base64.b64encode(b"fake").decode()

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
    err_spec = ModelSpec(
        name="smoke-error",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_error",
        backend_cls=ErrorTimeSeriesBackend,  # cloudpickled to worker
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    emb_spec = ModelSpec(
        name="smoke-embedding",
        model_type=ModelType.EMBEDDING,
        backend="_smoke_embedding",
        backend_cls=SmokeEmbeddingBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    seg_spec = ModelSpec(
        name="smoke-segmentation",
        model_type=ModelType.SEGMENTATION,
        backend="_smoke_segmentation",
        backend_cls=SmokeSegmentationBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    mol_spec = ModelSpec(
        name="smoke-molecular",
        model_type=ModelType.MOLECULAR,
        backend="_smoke_molecular",
        backend_cls=SmokeMolecularBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    depth_spec = ModelSpec(
        name="smoke-depth",
        model_type=ModelType.DEPTH,
        backend="_smoke_depth",
        backend_cls=SmokeDepthBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    det_spec = ModelSpec(
        name="smoke-detection",
        model_type=ModelType.DETECTION,
        backend="_smoke_detection",
        backend_cls=SmokeDetectionBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    weather_spec = ModelSpec(
        name="smoke-weather",
        model_type=ModelType.WEATHER,
        backend="_smoke_weather",
        backend_cls=SmokeWeatherBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )

    server = ModelServer(
        models=[
            cls_spec,
            reg_spec,
            err_spec,
            emb_spec,
            seg_spec,
            mol_spec,
            depth_spec,
            det_spec,
            weather_spec,
        ],
        host="127.0.0.1",
        port=_SMOKE_PORT,
    )
    server.run()

    base_cls = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-ts"
    base_reg = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-registry"
    base_err = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-error"

    for label, url in [
        ("smoke-ts", base_cls),
        ("smoke-registry", base_reg),
        ("smoke-error", base_err),
        ("smoke-embedding", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-embedding"),
        ("smoke-segmentation", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-segmentation"),
        ("smoke-molecular", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-molecular"),
        ("smoke-depth", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-depth"),
        ("smoke-detection", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-detection"),
        ("smoke-weather", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-weather"),
    ]:
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

    yield {
        "cls": base_cls,
        "registry": base_reg,
        "error": base_err,
        "embedding": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-embedding",
        "segmentation": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-segmentation",
        "molecular": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-molecular",
        "depth": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-depth",
        "detection": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-detection",
        "weather": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-weather",
        "server": server,
    }

    server.shutdown()
    ray.shutdown()


# Convenience fixtures that unpack from the combined urls fixture
@pytest.fixture(scope="module")
def serving_url(urls: dict) -> str:
    return urls["cls"]


@pytest.fixture(scope="module")
def registry_serving_url(urls: dict) -> str:
    return urls["registry"]


@pytest.fixture(scope="module")
def error_serving_url(urls: dict) -> str:
    return urls["error"]


@pytest.fixture(scope="module")
def model_server(urls: dict):  # type: ignore[type-arg]
    return urls["server"]


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


def test_backend_error_returns_500(error_serving_url: str) -> None:
    """A backend that raises must return HTTP 500 with a structured detail field.

    Verifies the service boundary error handling: the exception is caught in
    predict(), logged server-side, and returned as a JSON error response rather
    than crashing the actor or returning an opaque 500.
    """
    payload = {
        "model_type": "time_series",
        "model_name": "smoke-error",
        "history": [1.0, 2.0],
        "horizon": 2,
        "frequency": "1h",
    }
    r = requests.post(f"{error_serving_url}/predict", json=payload)
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "RuntimeError" in detail
    assert "backend exploded" in detail


# ---------------------------------------------------------------------------
# New-modality end-to-end smoke tests
# ---------------------------------------------------------------------------


def test_embedding_predict(urls: dict) -> None:
    """EmbeddingRequest routes correctly through the AnyRequest union."""
    url = urls["embedding"]
    payload = {
        "model_type": "embedding",
        "model_name": "smoke-embedding",
        "texts": ["hello", "world"],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 3
    assert len(body["embeddings"]) == 2


def test_segmentation_predict(urls: dict) -> None:
    """SegmentationRequest routes correctly; mask_b64 returned."""
    url = urls["segmentation"]
    payload = {
        "model_type": "segmentation",
        "model_name": "smoke-segmentation",
        "image_b64": _FAKE_IMG_B64,
        "box": [0.0, 0.0, 64.0, 64.0],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["height"] == 4
    assert body["width"] == 4
    assert len(body["masks_b64"]) == 1
    assert len(body["scores"]) == 1


def test_molecular_predict(urls: dict) -> None:
    """MolecularRequest routes correctly; embeddings returned for each sequence."""
    url = urls["molecular"]
    payload = {
        "model_type": "molecular",
        "model_name": "smoke-molecular",
        "sequences": ["MKTII", "ACDEF"],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 3
    assert len(body["embeddings"]) == 2


def test_depth_predict(urls: dict) -> None:
    """DepthRequest routes correctly; depth_b64 and bounds returned."""
    url = urls["depth"]
    payload = {
        "model_type": "depth",
        "model_name": "smoke-depth",
        "image_b64": _FAKE_IMG_B64,
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["height"] == 4
    assert body["width"] == 4
    assert "depth_b64" in body
    assert body["min_depth"] == pytest.approx(0.5)
    assert body["max_depth"] == pytest.approx(0.5)


def test_detection_predict(urls: dict) -> None:
    """DetectionRequest routes correctly; boxes/labels/scores returned."""
    url = urls["detection"]
    payload = {
        "model_type": "detection",
        "model_name": "smoke-detection",
        "image_b64": _FAKE_IMG_B64,
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["boxes"]) == 1
    assert body["labels"] == ["cat"]
    assert body["scores"] == pytest.approx([0.9])
    assert body["width"] == 640
    assert body["height"] == 480


def test_weather_predict(urls: dict) -> None:
    """WeatherRequest routes correctly; surface and atmospheric forecasts returned."""
    import base64

    import numpy as np

    n_lat, n_lon, n_lev = 4, 8, 3
    lat = [90.0, 60.0, 30.0, 0.0]
    lon = [float(i * 45) for i in range(n_lon)]
    levels = [1000, 500, 100]

    def _enc(arr: np.ndarray) -> str:
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

    zeros_surf = _enc(np.zeros((n_lat, n_lon), dtype=np.float32))
    zeros_atmos = _enc(np.zeros((n_lev, n_lat, n_lon), dtype=np.float32))

    url = urls["weather"]
    payload = {
        "model_type": "weather",
        "model_name": "smoke-weather",
        "surface_vars": {"2m_temperature": zeros_surf},
        "atmospheric_vars": {"temperature": zeros_atmos},
        "prev_surface_vars": {"2m_temperature": zeros_surf},
        "prev_atmospheric_vars": {"temperature": zeros_atmos},
        "lat": lat,
        "lon": lon,
        "pressure_levels": levels,
        "current_time": "2023-01-01T12:00:00",
        "n_steps": 2,
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_steps"] == 2
    assert body["step_hours"] == 6
    assert len(body["surface_forecasts"]) == 2
    assert len(body["atmospheric_forecasts"]) == 2
    assert body["forecast_times"][0] == "2023-01-01T18:00:00"
    assert body["forecast_times"][1] == "2023-01-02T00:00:00"
    assert "2m_temperature" in body["surface_forecasts"][0]
    assert "temperature" in body["atmospheric_forecasts"][0]


def test_wrong_model_type_for_embedding_returns_422(urls: dict) -> None:
    """Sending time_series payload to an embedding deployment → 422."""
    url = urls["embedding"]
    payload = {
        "model_type": "time_series",
        "model_name": "smoke-embedding",
        "history": [1.0, 2.0],
        "horizon": 2,
        "frequency": "1h",
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 422


def test_hot_swap(serving_url: str, model_server) -> None:  # type: ignore[type-arg]
    """server.update() replaces a deployment in place with no URL change.

    This test mutates smoke-ts and must run last in the module.

    Flow:
      1. Verify smoke-ts currently returns mean=[0.42, 0.42]
      2. Call server.update() with a spec pointing to the 0.99 backend
      3. Poll until predictions return 0.99 (rolling update complete)
    """
    from tests.stubs import SmokeTimeSeriesRegistryBackend

    payload = {
        "model_type": "time_series",
        "model_name": "smoke-ts",
        "history": [1.0, 2.0],
        "horizon": 2,
        "frequency": "1h",
    }

    # Confirm baseline before swap
    r = requests.post(f"{serving_url}/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["mean"] == [0.42, 0.42]

    # Swap to a backend that returns 0.99
    new_spec = ModelSpec(
        name="smoke-ts",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts_registry",
        backend_cls=SmokeTimeSeriesRegistryBackend,
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )
    model_server.update(new_spec)

    # Poll until the new backend is serving (rolling update may take a moment)
    deadline = time.time() + 30
    last_response = None
    while time.time() < deadline:
        r = requests.post(f"{serving_url}/predict", json=payload)
        if r.status_code == 200 and r.json().get("mean") == [0.99, 0.99]:
            last_response = r
            break
        time.sleep(0.5)
    else:
        pytest.fail("Hot-swap did not complete within 30s")

    assert last_response is not None
    assert last_response.json()["mean"] == [0.99, 0.99]
