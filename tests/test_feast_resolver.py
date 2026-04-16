"""Tests for FeastResolver and Feast integration in _build_asgi_app.

FeastResolver is tested with a mocked feast.FeatureStore — no Feast
installation or online store is required.

_build_asgi_app Feast integration is tested via starlette TestClient,
mirroring the pattern in test_modal_server.py.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

import tests.stubs  # noqa: F401 — registers _smoke_ts
from sheaf.api.base import ModelType
from sheaf.api.time_series import FeatureRef
from sheaf.integrations.feast import FeastResolver
from sheaf.modal_server import _build_asgi_app
from sheaf.spec import ModelSpec, ResourceConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_spec(name: str = "ts", feast_repo_path: str | None = None) -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        resources=ResourceConfig(num_cpus=1),
        feast_repo_path=feast_repo_path,
    )


def _make_mock_store(feature_name: str, values: list) -> MagicMock:
    """Return a MagicMock FeatureStore that returns `values` for `feature_name`."""
    store = MagicMock()
    store.get_online_features.return_value.to_dict.return_value = {
        feature_name: [values]
    }
    return store


def _make_fake_feast(store: MagicMock) -> types.ModuleType:
    mod = types.ModuleType("feast")
    mod.FeatureStore = MagicMock(return_value=store)  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# FeastResolver unit tests
# ---------------------------------------------------------------------------


class TestFeastResolver:
    def test_load_initialises_feature_store(self) -> None:
        mock_store = MagicMock()
        fake_feast = _make_fake_feast(mock_store)
        with patch.dict(sys.modules, {"feast": fake_feast}):
            r = FeastResolver("/repo/path")
            r.load()
        fake_feast.FeatureStore.assert_called_once_with(repo_path="/repo/path")
        assert r._store is mock_store

    def test_load_raises_without_feast(self) -> None:
        with patch.dict(sys.modules, {"feast": None}):  # type: ignore[dict-item]
            r = FeastResolver("/repo")
            with pytest.raises(ImportError, match="feast is required"):
                r.load()

    def test_resolve_calls_get_online_features(self) -> None:
        mock_store = _make_mock_store("close_history", [1.0, 2.0, 3.0])
        r = FeastResolver("/repo")
        r._store = mock_store

        ref = FeatureRef(
            feature_view="asset_prices",
            feature_name="close_history",
            entity_key="ticker",
            entity_value="AAPL",
        )
        result = r.resolve(ref)

        mock_store.get_online_features.assert_called_once_with(
            features=["asset_prices:close_history"],
            entity_rows=[{"ticker": "AAPL"}],
        )
        assert result == [1.0, 2.0, 3.0]

    def test_resolve_coerces_ints_to_float(self) -> None:
        mock_store = _make_mock_store("hist", [1, 2, 3])
        r = FeastResolver("/repo")
        r._store = mock_store

        ref = FeatureRef(
            feature_view="v", feature_name="hist", entity_key="k", entity_value="x"
        )
        result = r.resolve(ref)
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(v, float) for v in result)

    def test_resolve_raises_if_feature_missing(self) -> None:
        mock_store = MagicMock()
        mock_store.get_online_features.return_value.to_dict.return_value = {
            "other_feature": [[1.0]]
        }
        r = FeastResolver("/repo")
        r._store = mock_store

        ref = FeatureRef(
            feature_view="v",
            feature_name="close_history",
            entity_key="k",
            entity_value="x",
        )
        with pytest.raises(ValueError, match="not found"):
            r.resolve(ref)

    def test_resolve_raises_if_not_list(self) -> None:
        mock_store = _make_mock_store("close", 42.0)  # scalar, not a list
        r = FeastResolver("/repo")
        r._store = mock_store

        ref = FeatureRef(
            feature_view="v", feature_name="close", entity_key="k", entity_value="x"
        )
        with pytest.raises(ValueError, match="must return list"):
            r.resolve(ref)

    def test_resolve_raises_before_load(self) -> None:
        r = FeastResolver("/repo")  # no load() called
        ref = FeatureRef(
            feature_view="v", feature_name="f", entity_key="k", entity_value="x"
        )
        with pytest.raises(RuntimeError, match="load()"):
            r.resolve(ref)

    def test_resolve_uses_entity_key_and_value(self) -> None:
        mock_store = _make_mock_store("f", [0.1])
        r = FeastResolver("/repo")
        r._store = mock_store

        ref = FeatureRef(
            feature_view="sales",
            feature_name="f",
            entity_key="store_id",
            entity_value="store_42",
        )
        r.resolve(ref)

        _, kwargs = mock_store.get_online_features.call_args
        assert kwargs["entity_rows"] == [{"store_id": "store_42"}]


# ---------------------------------------------------------------------------
# _build_asgi_app Feast integration tests (via TestClient)
# ---------------------------------------------------------------------------


class TestBuildAsgiAppFeast:
    def _ts_payload(self) -> dict:
        return {
            "model_type": "time_series",
            "model_name": "ts",
            "history": [1.0, 2.0, 3.0],
            "horizon": 3,
            "frequency": "1h",
        }

    def _feature_ref_payload(self, feature_value: list[float] | None = None) -> dict:
        return {
            "model_type": "time_series",
            "model_name": "ts",
            "feature_ref": {
                "feature_view": "asset_prices",
                "feature_name": "close_history",
                "entity_key": "ticker",
                "entity_value": "AAPL",
            },
            "horizon": 3,
            "frequency": "1h",
        }

    def test_feature_ref_resolves_and_predicts(self) -> None:
        resolved = [10.0, 20.0, 30.0]
        mock_store = _make_mock_store("close_history", resolved)
        fake_feast = _make_fake_feast(mock_store)

        with patch.dict(sys.modules, {"feast": fake_feast}):
            app = _build_asgi_app([_ts_spec("ts", feast_repo_path="/repo")])

        client = TestClient(app)
        r = client.post("/ts/predict", json=self._feature_ref_payload())
        assert r.status_code == 200, r.text
        assert r.json()["mean"] == [0.42, 0.42, 0.42]

    def test_feature_ref_passes_correct_entity_to_feast(self) -> None:
        resolved = [1.0, 2.0]
        mock_store = _make_mock_store("close_history", resolved)
        fake_feast = _make_fake_feast(mock_store)

        with patch.dict(sys.modules, {"feast": fake_feast}):
            app = _build_asgi_app([_ts_spec("ts", feast_repo_path="/repo")])

        client = TestClient(app)
        client.post("/ts/predict", json=self._feature_ref_payload())

        mock_store.get_online_features.assert_called_once_with(
            features=["asset_prices:close_history"],
            entity_rows=[{"ticker": "AAPL"}],
        )

    def test_feature_ref_without_feast_repo_returns_422(self) -> None:
        app = _build_asgi_app([_ts_spec("ts", feast_repo_path=None)])
        client = TestClient(app)
        r = client.post("/ts/predict", json=self._feature_ref_payload())
        assert r.status_code == 422
        assert "feast_repo_path" in r.json()["detail"]

    def test_feast_resolution_error_returns_502(self) -> None:
        mock_store = MagicMock()
        mock_store.get_online_features.side_effect = RuntimeError("Redis unavailable")
        fake_feast = _make_fake_feast(mock_store)

        with patch.dict(sys.modules, {"feast": fake_feast}):
            app = _build_asgi_app([_ts_spec("ts", feast_repo_path="/repo")])

        client = TestClient(app, raise_server_exceptions=False)
        r = client.post("/ts/predict", json=self._feature_ref_payload())
        assert r.status_code == 502
        assert "Feast resolution failed" in r.json()["detail"]
        assert "Redis unavailable" in r.json()["detail"]

    def test_raw_history_still_works_alongside_feast(self) -> None:
        """Feast configured but request sends raw history — should just work."""
        mock_store = _make_mock_store("f", [1.0])
        fake_feast = _make_fake_feast(mock_store)

        with patch.dict(sys.modules, {"feast": fake_feast}):
            app = _build_asgi_app([_ts_spec("ts", feast_repo_path="/repo")])

        client = TestClient(app)
        r = client.post("/ts/predict", json=self._ts_payload())
        assert r.status_code == 200
        mock_store.get_online_features.assert_not_called()
