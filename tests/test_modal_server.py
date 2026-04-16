"""Tests for ModalServer and _build_asgi_app.

_build_asgi_app is tested directly (no Modal infra needed) via starlette's
TestClient.  ModalServer is tested with a fake modal module injected into
sys.modules so no Modal credentials are required.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from sheaf.api.base import ModelType
from sheaf.modal_server import ModalServer, _build_asgi_app
from sheaf.spec import ModelSpec, ResourceConfig

# ---------------------------------------------------------------------------
# Fake modal module (no Modal SDK installed in dev venv)
# ---------------------------------------------------------------------------


def _make_fake_modal() -> types.ModuleType:
    mod = types.ModuleType("modal")

    class FakeApp:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **kwargs: object) -> object:
            def decorator(fn: object) -> object:
                return fn

            return decorator

    def asgi_app() -> object:
        def decorator(fn: object) -> object:
            return fn

        return decorator

    class _FakeImageInner:
        def pip_install(self, *args: object, **kwargs: object) -> _FakeImageInner:
            return self

    class FakeImage:
        def debian_slim(self, **kwargs: object) -> _FakeImageInner:
            return _FakeImageInner()

    mod.App = FakeApp  # type: ignore[attr-defined]
    mod.asgi_app = asgi_app  # type: ignore[attr-defined]
    mod.Image = FakeImage()  # type: ignore[attr-defined]

    return mod


_FAKE_MODAL = _make_fake_modal()

# ---------------------------------------------------------------------------
# Helpers — reuse smoke stubs (already registered in the registry)
# ---------------------------------------------------------------------------

import tests.stubs  # noqa: E402, F401 — registers _smoke_ts, _smoke_embedding, etc.


def _ts_spec(name: str = "ts") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )


def _emb_spec(name: str = "emb") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.EMBEDDING,
        backend="_smoke_embedding",
        resources=ResourceConfig(num_cpus=1, replicas=1),
    )


# ---------------------------------------------------------------------------
# _build_asgi_app tests
# ---------------------------------------------------------------------------


class TestBuildAsgiApp:
    def test_health_returns_ok(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        r = client.get("/ts/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_ready_returns_model_name(self) -> None:
        app = _build_asgi_app([_ts_spec("my-forecaster")])
        client = TestClient(app)
        r = client.get("/my-forecaster/ready")
        assert r.status_code == 200
        assert r.json() == {"status": "ready", "model": "my-forecaster"}

    def test_unknown_name_health_returns_404(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        r = client.get("/no-such-model/health")
        assert r.status_code == 404

    def test_unknown_name_predict_returns_404(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        payload = {
            "model_type": "time_series",
            "model_name": "ts",
            "history": [1.0, 2.0, 3.0],
            "horizon": 2,
            "frequency": "1h",
        }
        r = client.post("/no-such-model/predict", json=payload)
        assert r.status_code == 404

    def test_predict_time_series(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        payload = {
            "model_type": "time_series",
            "model_name": "ts",
            "history": [1.0, 2.0, 3.0, 4.0, 5.0],
            "horizon": 3,
            "frequency": "1h",
        }
        r = client.post("/ts/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["mean"] == [0.42, 0.42, 0.42]
        assert body["horizon"] == 3

    def test_predict_embedding(self) -> None:
        app = _build_asgi_app([_emb_spec()])
        client = TestClient(app)
        payload = {
            "model_type": "embedding",
            "model_name": "emb",
            "texts": ["hello", "world"],
        }
        r = client.post("/emb/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["dim"] == 3
        assert len(body["embeddings"]) == 2

    def test_wrong_model_type_returns_422(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        # Send a tabular payload to the time-series deployment
        payload = {
            "model_type": "tabular",
            "model_name": "ts",
            "context_X": [[1.0]],
            "context_y": [0],
            "query_X": [[2.0]],
        }
        r = client.post("/ts/predict", json=payload)
        assert r.status_code == 422

    def test_malformed_payload_returns_422(self) -> None:
        app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        r = client.post("/ts/predict", json={"model_type": "time_series"})
        assert r.status_code == 422

    def test_backend_exception_returns_500(self) -> None:
        app = _build_asgi_app(
            [
                ModelSpec(
                    name="err",
                    model_type=ModelType.TIME_SERIES,
                    backend="_smoke_error",
                    resources=ResourceConfig(num_cpus=1),
                )
            ]
        )
        client = TestClient(app, raise_server_exceptions=False)
        payload = {
            "model_type": "time_series",
            "model_name": "err",
            "history": [1.0, 2.0],
            "horizon": 2,
            "frequency": "1h",
        }
        r = client.post("/err/predict", json=payload)
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert "RuntimeError" in detail
        assert "backend exploded" in detail

    def test_unknown_backend_raises_at_startup(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            _build_asgi_app(
                [
                    ModelSpec(
                        name="x",
                        model_type=ModelType.TIME_SERIES,
                        backend="_no_such_backend",
                        resources=ResourceConfig(num_cpus=1),
                    )
                ]
            )

    def test_multiple_backends_same_app(self) -> None:
        app = _build_asgi_app([_ts_spec("ts"), _emb_spec("emb")])
        client = TestClient(app)

        r = client.get("/ts/health")
        assert r.status_code == 200

        r = client.get("/emb/health")
        assert r.status_code == 200

        r = client.post(
            "/ts/predict",
            json={
                "model_type": "time_series",
                "model_name": "ts",
                "history": [1.0],
                "horizon": 1,
                "frequency": "1h",
            },
        )
        assert r.status_code == 200
        assert r.json()["mean"] == [0.42]

        r = client.post(
            "/emb/predict",
            json={
                "model_type": "embedding",
                "model_name": "emb",
                "texts": ["hi"],
            },
        )
        assert r.status_code == 200
        assert r.json()["dim"] == 3

    def test_extra_backends_env_var_loaded(self) -> None:
        """SHEAF_EXTRA_BACKENDS is honoured by _build_asgi_app."""
        with patch.dict("os.environ", {"SHEAF_EXTRA_BACKENDS": "tests.stubs"}):
            app = _build_asgi_app([_ts_spec()])
        client = TestClient(app)
        r = client.get("/ts/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# ModalServer tests (fake modal module injected)
# ---------------------------------------------------------------------------


class TestModalServer:
    def test_init_raises_without_modal(self) -> None:
        with patch.dict(sys.modules, {"modal": None}):  # type: ignore[dict-item]
            with pytest.raises(ImportError, match="modal is required"):
                ModalServer(models=[_ts_spec()])

    def test_init_creates_app_attribute(self) -> None:
        with patch.dict(sys.modules, {"modal": _FAKE_MODAL}):
            server = ModalServer(models=[_ts_spec()], app_name="test-sheaf")
        assert server.app is not None
        assert server.app.name == "test-sheaf"

    def test_init_stores_models(self) -> None:
        spec = _ts_spec("my-model")
        with patch.dict(sys.modules, {"modal": _FAKE_MODAL}):
            server = ModalServer(models=[spec])
        assert server._models == [spec]

    def test_init_accepts_custom_image(self) -> None:
        fake_image = MagicMock()
        with patch.dict(sys.modules, {"modal": _FAKE_MODAL}):
            server = ModalServer(models=[_ts_spec()], image=fake_image)
        assert server.app is not None

    def test_app_name_propagated(self) -> None:
        with patch.dict(sys.modules, {"modal": _FAKE_MODAL}):
            server = ModalServer(models=[_ts_spec()], app_name="custom-name")
        assert server.app.name == "custom-name"

    def test_default_app_name_is_sheaf(self) -> None:
        with patch.dict(sys.modules, {"modal": _FAKE_MODAL}):
            server = ModalServer(models=[_ts_spec()])
        assert server.app.name == "sheaf"
