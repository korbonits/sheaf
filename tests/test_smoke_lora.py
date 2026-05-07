"""End-to-end Ray Serve smoke test for LoRA adapter multiplexing.

Spins up a real Ray cluster + Serve instance with ``_smoke_lora_diffusion``
(a stub backend that opts in to LoRA and records adapter calls), exercises
the full HTTP surface, and tears down.

Run explicitly:
    SHEAF_SMOKE_TEST=1 uv run pytest tests/test_smoke_lora.py -v -s

Skipped in normal CI because it starts a local Ray cluster (~5s overhead).
Set SHEAF_SMOKE_TEST=1 to enable.

What this verifies that the unit + ASGI tests can't:
  - load_adapters() actually fires inside a Ray Serve worker process
    (not the test process), surviving cloudpickle and runtime_env
  - The request routes through @serve.batch + bucket-by-resolved-adapter
  - 422 / 200 status codes are produced by the real Ray Serve HTTP layer
  - Concurrent requests with different adapter selections don't crash
"""

from __future__ import annotations

import base64
import os
import time

import pytest
import requests

from sheaf.api.base import ModelType
from sheaf.lora import LoRAAdapter, LoRAConfig
from sheaf.spec import ModelSpec, ResourceConfig
from tests.stubs import SmokeLoRADiffusionBackend

pytestmark = pytest.mark.skipif(
    not os.environ.get("SHEAF_SMOKE_TEST"),
    reason="Set SHEAF_SMOKE_TEST=1 to run Ray Serve smoke tests",
)

_SMOKE_PORT = 8788  # avoid clashing with test_smoke_ray (8787)


@pytest.fixture(scope="module")
def lora_url() -> str:
    """Bring up one Ray cluster + ModelServer with a LoRA-configured deployment."""
    import ray

    from sheaf.server import ModelServer

    ray.init(
        runtime_env={
            "working_dir": ".",
            "env_vars": {"SHEAF_EXTRA_BACKENDS": "tests.stubs"},
        }
    )

    spec = ModelSpec(
        name="smoke-lora",
        model_type=ModelType.DIFFUSION,
        backend="_smoke_lora_diffusion",
        backend_cls=SmokeLoRADiffusionBackend,
        resources=ResourceConfig(num_cpus=0.1, replicas=1),
        lora=LoRAConfig(
            adapters={
                "sketch": LoRAAdapter(source="/tmp/sketch.safetensors", weight=0.8),
                "watercolor": LoRAAdapter(source="hf:user/watercolor", weight=1.0),
            },
            default="sketch",
        ),
    )

    server = ModelServer(models=[spec], host="127.0.0.1", port=_SMOKE_PORT)
    server.run()

    base = f"http://127.0.0.1:{_SMOKE_PORT}/smoke-lora"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if requests.get(f"{base}/health", timeout=2).status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        server.shutdown()
        ray.shutdown()
        pytest.fail("smoke-lora did not become ready within 30s")

    yield base

    server.shutdown()
    ray.shutdown()


def _payload(adapters: list[str] | None = None, weights: list[float] | None = None):
    body: dict = {
        "model_type": "diffusion",
        "model_name": "smoke-lora",
        "prompt": "a cat on the moon",
        "height": 64,
        "width": 64,
        "seed": 1,
    }
    if adapters is not None:
        body["adapters"] = adapters
    if weights is not None:
        body["adapter_weights"] = weights
    return body


def _is_png(image_b64: str) -> bool:
    return base64.b64decode(image_b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_health(lora_url: str) -> None:
    r = requests.get(f"{lora_url}/health")
    assert r.status_code == 200


def test_predict_with_default_adapter(lora_url: str) -> None:
    """Empty `adapters` falls back to spec.lora.default ('sketch')."""
    r = requests.post(f"{lora_url}/predict", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert _is_png(body["image_b64"])
    assert body["height"] == 64


def test_predict_explicit_single_adapter(lora_url: str) -> None:
    r = requests.post(f"{lora_url}/predict", json=_payload(adapters=["watercolor"]))
    assert r.status_code == 200, r.text
    assert _is_png(r.json()["image_b64"])


def test_predict_fusion_with_explicit_weights(lora_url: str) -> None:
    r = requests.post(
        f"{lora_url}/predict",
        json=_payload(adapters=["sketch", "watercolor"], weights=[0.5, 0.7]),
    )
    assert r.status_code == 200, r.text


def test_predict_unknown_adapter_returns_422(lora_url: str) -> None:
    r = requests.post(f"{lora_url}/predict", json=_payload(adapters=["bogus"]))
    assert r.status_code == 422
    assert "Unknown adapter" in r.json()["detail"]


def test_predict_concurrent_mixed_adapters(lora_url: str) -> None:
    """Fire 8 concurrent requests with mixed adapter selections.

    Exercises bucket-by-resolved-adapter inside @serve.batch: the deployment
    must group these into homogeneous sub-batches and call set_active_adapters
    once per bucket without races.
    """
    import concurrent.futures

    payloads = [
        _payload(),  # default → sketch
        _payload(adapters=["watercolor"]),
        _payload(),  # default → sketch
        _payload(adapters=["watercolor"]),
        _payload(adapters=["sketch", "watercolor"], weights=[0.5, 0.5]),
        _payload(adapters=["sketch"]),
        _payload(adapters=["watercolor"]),
        _payload(adapters=["sketch", "watercolor"], weights=[0.5, 0.5]),
    ]

    def _call(p: dict) -> int:
        return requests.post(f"{lora_url}/predict", json=p).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(_call, payloads))

    assert all(s == 200 for s in statuses), statuses
