"""LoRA wire-up tests for modal_server._build_asgi_app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import tests.stubs  # noqa: F401 — registers _smoke_lora_diffusion + _smoke_diffusion
from sheaf.api.base import ModelType
from sheaf.lora import LoRAAdapter, LoRAConfig
from sheaf.modal_server import _build_asgi_app
from sheaf.spec import ModelSpec, ResourceConfig
from tests.stubs import SmokeLoRADiffusionBackend


def _spec(name: str = "flux", *, lora: LoRAConfig | None = None) -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.DIFFUSION,
        backend="_smoke_lora_diffusion",
        resources=ResourceConfig(num_cpus=1),
        lora=lora,
    )


def _payload(adapters: list[str] | None = None, weights: list[float] | None = None):
    body: dict = {
        "model_type": "diffusion",
        "model_name": "flux",
        "prompt": "a cat",
    }
    if adapters is not None:
        body["adapters"] = adapters
    if weights is not None:
        body["adapter_weights"] = weights
    return body


@pytest.fixture(autouse=True)
def _reset() -> None:
    SmokeLoRADiffusionBackend.reset()


class TestModalServerLoRA:
    def test_load_adapters_called_at_startup(self) -> None:
        cfg = LoRAConfig(
            adapters={
                "sketch": LoRAAdapter(source="/tmp/sketch", weight=0.8),
                "wc": LoRAAdapter(source="hf:user/wc"),
            },
            default="sketch",
        )
        _build_asgi_app([_spec(lora=cfg)])
        assert len(SmokeLoRADiffusionBackend.loaded_adapters) == 1
        loaded = SmokeLoRADiffusionBackend.loaded_adapters[0]
        assert set(loaded) == {"sketch", "wc"}

    def test_predict_with_default_adapter_calls_set(self) -> None:
        cfg = LoRAConfig(
            adapters={"sketch": LoRAAdapter(source="/tmp/sketch", weight=0.8)},
            default="sketch",
        )
        app = _build_asgi_app([_spec(lora=cfg)])
        client = TestClient(app)
        r = client.post("/flux/predict", json=_payload())
        assert r.status_code == 200
        # set_active_adapters fired with the default
        assert (["sketch"], [0.8]) in SmokeLoRADiffusionBackend.set_calls

    def test_predict_explicit_adapter_overrides_default(self) -> None:
        cfg = LoRAConfig(
            adapters={
                "sketch": LoRAAdapter(source="/tmp/sketch", weight=0.8),
                "wc": LoRAAdapter(source="/tmp/wc", weight=1.0),
            },
            default="sketch",
        )
        app = _build_asgi_app([_spec(lora=cfg)])
        client = TestClient(app)
        r = client.post("/flux/predict", json=_payload(adapters=["wc"]))
        assert r.status_code == 200
        assert (["wc"], [1.0]) in SmokeLoRADiffusionBackend.set_calls

    def test_predict_explicit_weights_win(self) -> None:
        cfg = LoRAConfig(
            adapters={"sketch": LoRAAdapter(source="/tmp/sketch", weight=0.8)},
        )
        app = _build_asgi_app([_spec(lora=cfg)])
        client = TestClient(app)
        r = client.post(
            "/flux/predict",
            json=_payload(adapters=["sketch"], weights=[0.3]),
        )
        assert r.status_code == 200
        assert (["sketch"], [0.3]) in SmokeLoRADiffusionBackend.set_calls

    def test_predict_unknown_adapter_returns_422(self) -> None:
        cfg = LoRAConfig(
            adapters={"sketch": LoRAAdapter(source="/tmp/sketch")},
        )
        app = _build_asgi_app([_spec(lora=cfg)])
        client = TestClient(app)
        r = client.post("/flux/predict", json=_payload(adapters=["nope"]))
        assert r.status_code == 422
        assert "Unknown adapter" in r.json()["detail"]
        # No set_active_adapters call should have happened
        assert SmokeLoRADiffusionBackend.set_calls == []

    def test_request_adapters_without_spec_lora_returns_422(self) -> None:
        app = _build_asgi_app([_spec(lora=None)])
        client = TestClient(app)
        r = client.post("/flux/predict", json=_payload(adapters=["sketch"]))
        assert r.status_code == 422
        assert "no LoRA configured" in r.json()["detail"]

    def test_no_lora_no_set_call(self) -> None:
        cfg = LoRAConfig(
            adapters={"sketch": LoRAAdapter(source="/tmp/sketch")}, default=None
        )
        app = _build_asgi_app([_spec(lora=cfg)])
        client = TestClient(app)
        r = client.post("/flux/predict", json=_payload())
        assert r.status_code == 200
        # No default + no request adapters → set_active_adapters not called.
        assert SmokeLoRADiffusionBackend.set_calls == []

    def test_unsupported_backend_with_lora_raises_at_startup(self) -> None:
        cfg = LoRAConfig(
            adapters={"sketch": LoRAAdapter(source="/tmp/sketch")},
        )
        spec = ModelSpec(
            name="flux",
            model_type=ModelType.DIFFUSION,
            backend="_smoke_diffusion",  # does not support LoRA
            resources=ResourceConfig(num_cpus=1),
            lora=cfg,
        )
        with pytest.raises(ValueError, match="does not support"):
            _build_asgi_app([spec])
