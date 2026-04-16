"""Quick live test for ModalServer — no GPU, no ML weights.

Deploys two inline stub backends via ModalServer and hits the live
HTTP endpoints to verify the serving layer works on Modal.

Stubs are passed via backend_cls so no extra files need to be mounted.
The closure (including the classes) is cloudpickled via serialized=True.

Run:
    modal run examples/test_modal_server_live.py
"""

from __future__ import annotations

import time

import modal
import requests

from sheaf import ModalServer
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.spec import ModelSpec, ResourceConfig

# ---------------------------------------------------------------------------
# Inline stub backends — no weights, no registry lookup needed
# ---------------------------------------------------------------------------


class _StubTSBackend(ModelBackend):
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
            mean=[0.42] * request.horizon,
        )


class _StubEmbBackend(ModelBackend):
    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.EMBEDDING

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, EmbeddingRequest)
        n = len(request.texts or request.images_b64 or [])
        return EmbeddingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=[[1.0, 0.0, 0.0]] * n,
            dim=3,
        )


# ---------------------------------------------------------------------------
# ModalServer — minimal image, no ML deps
# ---------------------------------------------------------------------------

# Minimal image: no torch/ray — just the runtime needs for ModalServer itself.
# add_local_python_source overlays the local sheaf package so modal_server.py
# (not yet published to PyPI) is available in the container.
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]>=0.110.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "cloudpickle>=3.0.0",
    )
    .add_local_python_source("sheaf")
)

server = ModalServer(
    models=[
        ModelSpec(
            name="ts",
            model_type=ModelType.TIME_SERIES,
            backend="__stub_ts__",
            backend_cls=_StubTSBackend,
            resources=ResourceConfig(num_cpus=1),
        ),
        ModelSpec(
            name="emb",
            model_type=ModelType.EMBEDDING,
            backend="__stub_emb__",
            backend_cls=_StubEmbBackend,
            resources=ResourceConfig(num_cpus=1),
        ),
    ],
    app_name="sheaf-live-test",
    gpu=None,
    image=_image,
    min_containers=0,
)

app = server.app


@app.local_entrypoint()
def main() -> None:
    url = server._serve_fn.get_web_url()
    print("\n--- ModalServer live test ---")
    print(f"Web URL: {url}")

    # Wait for the ASGI app to be ready (cold-start can take up to 3 min)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/ts/health", timeout=15)
            if r.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
        time.sleep(3)
    else:
        raise TimeoutError("ASGI app did not become ready within 180s")

    # --- health ---
    for name in ("ts", "emb"):
        r = requests.get(f"{url}/{name}/health")
        assert r.status_code == 200, f"health {name}: {r.status_code}"
        assert r.json() == {"status": "ok"}
        print(f"  GET /{name}/health : OK")

    # --- ready ---
    r = requests.get(f"{url}/ts/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "model": "ts"}
    print("  GET /ts/ready      : OK")

    # --- time series predict ---
    r = requests.post(
        f"{url}/ts/predict",
        json={
            "model_type": "time_series",
            "model_name": "ts",
            "history": [1.0, 2.0, 3.0, 4.0, 5.0],
            "horizon": 3,
            "frequency": "1h",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mean"] == [0.42, 0.42, 0.42]
    print(f"  POST /ts/predict   : mean={body['mean']}")

    # --- embedding predict ---
    r = requests.post(
        f"{url}/emb/predict",
        json={
            "model_type": "embedding",
            "model_name": "emb",
            "texts": ["hello", "world"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == 3
    assert len(body["embeddings"]) == 2
    print(f"  POST /emb/predict  : dim={body['dim']}, n={len(body['embeddings'])}")

    # --- 422 on bad payload ---
    r = requests.post(f"{url}/ts/predict", json={"model_type": "time_series"})
    assert r.status_code == 422
    print("  POST /ts/predict (malformed)   : 422 OK")

    # --- 404 on unknown deployment ---
    r = requests.get(f"{url}/nope/health")
    assert r.status_code == 404
    print("  GET /nope/health (unknown)     : 404 OK")

    print("\nAll assertions passed. ModalServer end-to-end: OK")
