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
    SmokeAudioGenerationBackend,
    SmokeDepthBackend,
    SmokeDetectionBackend,
    SmokeDiffusionBackend,
    SmokeEmbeddingBackend,
    SmokeGenomicBackend,
    SmokeMaterialsBackend,
    SmokeMolecularBackend,
    SmokeMultimodalEmbeddingBackend,
    SmokeSatelliteBackend,
    SmokeSegmentationBackend,
    SmokeSmallMoleculeBackend,
    SmokeTabularBackend,
    SmokeTimeSeriesBackend,
    SmokeTTSBackend,
    SmokeVideoBackend,
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
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    reg_spec = ModelSpec(
        name="smoke-registry",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts_registry",  # string lookup — no backend_cls
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    err_spec = ModelSpec(
        name="smoke-error",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_error",
        backend_cls=ErrorTimeSeriesBackend,  # cloudpickled to worker
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    emb_spec = ModelSpec(
        name="smoke-embedding",
        model_type=ModelType.EMBEDDING,
        backend="_smoke_embedding",
        backend_cls=SmokeEmbeddingBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    seg_spec = ModelSpec(
        name="smoke-segmentation",
        model_type=ModelType.SEGMENTATION,
        backend="_smoke_segmentation",
        backend_cls=SmokeSegmentationBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    mol_spec = ModelSpec(
        name="smoke-molecular",
        model_type=ModelType.MOLECULAR,
        backend="_smoke_molecular",
        backend_cls=SmokeMolecularBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    depth_spec = ModelSpec(
        name="smoke-depth",
        model_type=ModelType.DEPTH,
        backend="_smoke_depth",
        backend_cls=SmokeDepthBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    det_spec = ModelSpec(
        name="smoke-detection",
        model_type=ModelType.DETECTION,
        backend="_smoke_detection",
        backend_cls=SmokeDetectionBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    weather_spec = ModelSpec(
        name="smoke-weather",
        model_type=ModelType.WEATHER,
        backend="_smoke_weather",
        backend_cls=SmokeWeatherBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    satellite_spec = ModelSpec(
        name="smoke-satellite",
        model_type=ModelType.GEOSPATIAL,
        backend="_smoke_satellite",
        backend_cls=SmokeSatelliteBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    genomic_spec = ModelSpec(
        name="smoke-genomic",
        model_type=ModelType.GENOMIC,
        backend="_smoke_genomic",
        backend_cls=SmokeGenomicBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    materials_spec = ModelSpec(
        name="smoke-materials",
        model_type=ModelType.MATERIALS,
        backend="_smoke_materials",
        backend_cls=SmokeMaterialsBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    small_mol_spec = ModelSpec(
        name="smoke-small-molecule",
        model_type=ModelType.SMALL_MOLECULE,
        backend="_smoke_small_molecule",
        backend_cls=SmokeSmallMoleculeBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    audio_gen_spec = ModelSpec(
        name="smoke-audio-generation",
        model_type=ModelType.AUDIO_GENERATION,
        backend="_smoke_audio_generation",
        backend_cls=SmokeAudioGenerationBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    multimodal_spec = ModelSpec(
        name="smoke-multimodal",
        model_type=ModelType.MULTIMODAL_EMBEDDING,
        backend="_smoke_multimodal",
        backend_cls=SmokeMultimodalEmbeddingBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    tabular_spec = ModelSpec(
        name="smoke-tabular",
        model_type=ModelType.TABULAR,
        backend="_smoke_tabular",
        backend_cls=SmokeTabularBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    tts_spec = ModelSpec(
        name="smoke-tts",
        model_type=ModelType.TTS,
        backend="_smoke_tts",
        backend_cls=SmokeTTSBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    diffusion_spec = ModelSpec(
        name="smoke-diffusion",
        model_type=ModelType.DIFFUSION,
        backend="_smoke_diffusion",
        backend_cls=SmokeDiffusionBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
    )
    video_spec = ModelSpec(
        name="smoke-video",
        model_type=ModelType.VIDEO,
        backend="_smoke_video",
        backend_cls=SmokeVideoBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
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
            satellite_spec,
            genomic_spec,
            materials_spec,
            small_mol_spec,
            audio_gen_spec,
            multimodal_spec,
            tabular_spec,
            tts_spec,
            diffusion_spec,
            video_spec,
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
        ("smoke-satellite", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-satellite"),
        ("smoke-genomic", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-genomic"),
        ("smoke-materials", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-materials"),
        (
            "smoke-small-molecule",
            f"http://127.0.0.1:{_SMOKE_PORT}/smoke-small-molecule",
        ),
        (
            "smoke-audio-generation",
            f"http://127.0.0.1:{_SMOKE_PORT}/smoke-audio-generation",
        ),
        (
            "smoke-multimodal",
            f"http://127.0.0.1:{_SMOKE_PORT}/smoke-multimodal",
        ),
        ("smoke-tabular", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-tabular"),
        ("smoke-tts", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-tts"),
        ("smoke-diffusion", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-diffusion"),
        ("smoke-video", f"http://127.0.0.1:{_SMOKE_PORT}/smoke-video"),
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
        "satellite": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-satellite",
        "genomic": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-genomic",
        "materials": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-materials",
        "small_molecule": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-small-molecule",
        "audio_generation": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-audio-generation",
        "multimodal": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-multimodal",
        "tabular": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-tabular",
        "tts": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-tts",
        "diffusion": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-diffusion",
        "video": f"http://127.0.0.1:{_SMOKE_PORT}/smoke-video",
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


def test_satellite_predict(urls: dict) -> None:
    """SatelliteRequest routes correctly; scene embedding returned."""
    import base64

    import numpy as np

    n_time, n_bands, h, w = 2, 6, 16, 16
    pixels = np.zeros((n_time, n_bands, h, w), dtype=np.float32)
    pixels_b64 = base64.b64encode(pixels.tobytes()).decode()

    url = urls["satellite"]
    payload = {
        "model_type": "geospatial",
        "model_name": "smoke-satellite",
        "pixels_b64": pixels_b64,
        "n_time": n_time,
        "n_bands": n_bands,
        "height": h,
        "width": w,
        "band_names": ["blue", "green", "red", "nir08", "swir16", "swir22"],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 4
    assert len(body["embedding"]) == 4
    assert body["n_time"] == n_time


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


def test_small_molecule_predict(urls: dict) -> None:
    """SmallMoleculeRequest routes correctly; embeddings returned per SMILES."""
    url = urls["small_molecule"]
    payload = {
        "model_type": "small_molecule",
        "model_name": "smoke-small-molecule",
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 4
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 4


def test_materials_predict(urls: dict) -> None:
    """MaterialsRequest routes correctly; energy + forces returned."""
    import numpy as np

    url = urls["materials"]
    # CO2: C at origin, two O atoms at ±1.16 Å
    atomic_numbers = [6, 8, 8]
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]], dtype=np.float32
    )
    positions_b64 = base64.b64encode(positions.tobytes()).decode()

    payload = {
        "model_type": "materials",
        "model_name": "smoke-materials",
        "atomic_numbers": atomic_numbers,
        "positions_b64": positions_b64,
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_atoms"] == 3
    assert body["energy"] == pytest.approx(-42.0)
    assert body["forces_b64"] is not None
    forces = np.frombuffer(
        base64.b64decode(body["forces_b64"]), dtype=np.float32
    ).reshape(3, 3)
    assert forces.shape == (3, 3)


def test_genomic_predict(urls: dict) -> None:
    """GenomicRequest routes correctly; embeddings returned for each sequence."""
    url = urls["genomic"]
    payload = {
        "model_type": "genomic",
        "model_name": "smoke-genomic",
        "sequences": ["ATCGATCG", "GCTAGCTA"],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 4
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 4


def test_audio_generation_predict(urls: dict) -> None:
    """AudioGenerationRequest routes correctly; WAV audio returned."""
    url = urls["audio_generation"]
    payload = {
        "model_type": "audio_generation",
        "model_name": "smoke-audio-generation",
        "prompt": "happy jazz piano",
        "duration_s": 5.0,
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sampling_rate"] == 32000
    assert body["duration_s"] > 0
    wav_bytes = base64.b64decode(body["audio_b64"])
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_multimodal_embedding_predict(urls: dict) -> None:
    """MultimodalEmbeddingRequest routes correctly; embeddings returned per text."""
    url = urls["multimodal"]
    payload = {
        "model_type": "multimodal_embedding",
        "model_name": "smoke-multimodal",
        "texts": ["a photo of a dog", "a photo of a cat"],
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 4
    assert body["modality"] == "text"
    assert len(body["embeddings"]) == 2
    assert len(body["embeddings"][0]) == 4


def test_tabular_predict(urls: dict) -> None:
    """TabularRequest routes correctly; predictions + probabilities returned."""
    url = urls["tabular"]
    payload = {
        "model_type": "tabular",
        "model_name": "smoke-tabular",
        "context_X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "context_y": [0, 1, 0],
        "query_X": [[2.0, 3.0], [4.0, 5.0]],
        "task": "classification",
        "output_mode": "probabilities",
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "classification"
    assert body["n_context"] == 3
    assert body["n_query"] == 2
    assert body["predictions"] == [0, 0]
    assert len(body["probabilities"]) == 2
    assert body["probabilities"][0] == pytest.approx([0.8, 0.2])
    assert body["classes"] == [0, 1]


def test_tts_predict(urls: dict) -> None:
    """TTSRequest routes correctly; WAV audio returned."""
    url = urls["tts"]
    payload = {
        "model_type": "tts",
        "model_name": "smoke-tts",
        "text": "Hello from sheaf.",
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_rate"] == 24000
    wav_bytes = base64.b64decode(body["audio_b64"])
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_diffusion_predict(urls: dict) -> None:
    """DiffusionRequest routes correctly; PNG image returned."""
    url = urls["diffusion"]
    payload = {
        "model_type": "diffusion",
        "model_name": "smoke-diffusion",
        "prompt": "a serene mountain landscape",
        "height": 64,
        "width": 64,
        "seed": 42,
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seed"] == 42
    assert body["height"] == 64
    assert body["width"] == 64
    png_bytes = base64.b64decode(body["image_b64"])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_video_predict_embedding(urls: dict) -> None:
    """VideoRequest (embedding) routes correctly; embedding vector returned."""
    url = urls["video"]
    frame_b64 = base64.b64encode(b"fake-frame").decode()
    payload = {
        "model_type": "video",
        "model_name": "smoke-video",
        "frames_b64": [frame_b64] * 4,
        "task": "embedding",
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "embedding"
    assert body["dim"] == 768
    assert len(body["embedding"]) == 768


def test_video_predict_classification(urls: dict) -> None:
    """VideoRequest (classification) routes correctly; labels and scores returned."""
    url = urls["video"]
    frame_b64 = base64.b64encode(b"fake-frame").decode()
    payload = {
        "model_type": "video",
        "model_name": "smoke-video",
        "frames_b64": [frame_b64] * 4,
        "task": "classification",
    }
    r = requests.post(f"{url}/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "classification"
    assert len(body["labels"]) == len(body["scores"])
    assert body["scores"][0] == pytest.approx(0.8)


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
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
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
