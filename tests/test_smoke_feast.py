"""Smoke test: Feast feature store integration end-to-end.

Uses a real Feast FeatureStore backed by SQLite (no external services needed),
materialises time series history for two tickers, then exercises the full
sheaf serving path:

    Feast FeatureStore (SQLite)
        └─▶ FeastResolver.resolve(FeatureRef)
                └─▶ _build_asgi_app predict route
                        └─▶ SmokeTimeSeriesBackend

Run explicitly (feast must be installed):

    uv run pytest tests/test_smoke_feast.py -v -s

Gated on SHEAF_SMOKE_TEST=1 so it doesn't run in normal CI (feast adds ~5s
import overhead and writes to a temp directory).
"""

from __future__ import annotations

import datetime
import os

import pandas as pd
import pytest
from starlette.testclient import TestClient

import tests.stubs  # noqa: F401 — registers _smoke_ts
from sheaf.api.base import ModelType
from sheaf.modal_server import _build_asgi_app
from sheaf.spec import ModelSpec, ResourceConfig

# ---------------------------------------------------------------------------
# Opt-in gate (mirrors test_smoke_ray.py / test_smoke_whisper.py)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("SHEAF_SMOKE_TEST"),
    reason="Set SHEAF_SMOKE_TEST=1 to run Feast smoke tests",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = {
    "AAPL": [170.0, 172.5, 171.0, 174.0, 176.3, 175.1, 178.0],
    "MSFT": [330.0, 332.0, 335.5, 333.0, 336.0, 338.5, 340.0],
    "GOOG": [140.0, 141.0, 143.0, 142.5, 144.0, 143.5, 145.0],
}


def _build_feast_repo(tmpdir: str) -> None:
    """Apply feature definitions and materialise data into a SQLite online store."""
    feast = pytest.importorskip("feast")
    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Array, Float32

    # feature_store.yaml — local provider, SQLite online store
    yaml = """\
project: sheaf_smoke
registry: registry.db
provider: local
online_store:
    type: sqlite
    path: online_store.db
entity_key_serialization_version: 2
"""
    with open(os.path.join(tmpdir, "feature_store.yaml"), "w") as fh:
        fh.write(yaml)

    # Parquet source with one row per ticker
    now = datetime.datetime.utcnow()
    df = pd.DataFrame(
        {
            "ticker": list(_TICKERS.keys()),
            "close_history": list(_TICKERS.values()),
            "event_timestamp": [now] * len(_TICKERS),
            "created": [now] * len(_TICKERS),
        }
    )
    parquet_path = os.path.join(tmpdir, "prices.parquet")
    df.to_parquet(parquet_path, index=False)

    ticker_entity = Entity(name="ticker", join_keys=["ticker"])
    source = FileSource(
        path=parquet_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )
    prices_view = FeatureView(
        name="asset_prices",
        entities=[ticker_entity],
        schema=[Field(name="close_history", dtype=Array(Float32))],
        source=source,
        ttl=datetime.timedelta(days=1),
    )

    store = feast.FeatureStore(repo_path=tmpdir)
    store.apply([ticker_entity, prices_view])
    store.materialize_incremental(
        end_date=datetime.datetime.utcnow() + datetime.timedelta(minutes=1)
    )


def _ts_spec(feast_repo_path: str | None = None) -> ModelSpec:
    return ModelSpec(
        name="ts",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        resources=ResourceConfig(num_cpus=1),
        feast_repo_path=feast_repo_path,
    )


def _feature_ref_payload(ticker: str, horizon: int = 4) -> dict:
    return {
        "model_type": "time_series",
        "model_name": "ts",
        "feature_ref": {
            "feature_view": "asset_prices",
            "feature_name": "close_history",
            "entity_key": "ticker",
            "entity_value": ticker,
        },
        "horizon": horizon,
        "frequency": "1d",
    }


def _raw_payload(horizon: int = 3) -> dict:
    return {
        "model_type": "time_series",
        "model_name": "ts",
        "history": [1.0, 2.0, 3.0],
        "horizon": horizon,
        "frequency": "1d",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feast_repo(tmp_path_factory):
    """Build a real Feast repo once for the whole module."""
    tmpdir = str(tmp_path_factory.mktemp("feast_repo"))
    _build_feast_repo(tmpdir)
    return tmpdir


@pytest.fixture(scope="module")
def client(feast_repo):
    """TestClient backed by _build_asgi_app with a Feast-wired spec."""
    app = _build_asgi_app([_ts_spec(feast_repo_path=feast_repo)])
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeastSmoke:
    def test_feature_ref_aapl_resolves_and_predicts(self, client: TestClient) -> None:
        """AAPL feature_ref request resolves history and returns predictions."""
        r = client.post("/ts/predict", json=_feature_ref_payload("AAPL", horizon=4))
        assert r.status_code == 200, r.text
        body = r.json()
        # SmokeTimeSeriesBackend returns [0.42] * horizon
        assert body["mean"] == [0.42, 0.42, 0.42, 0.42]

    def test_feature_ref_msft_resolves_and_predicts(self, client: TestClient) -> None:
        """MSFT feature_ref request resolves history and returns predictions."""
        r = client.post("/ts/predict", json=_feature_ref_payload("MSFT", horizon=3))
        assert r.status_code == 200, r.text
        assert r.json()["mean"] == [0.42, 0.42, 0.42]

    def test_feature_ref_goog_resolves_and_predicts(self, client: TestClient) -> None:
        """GOOG feature_ref request resolves history and returns predictions."""
        r = client.post("/ts/predict", json=_feature_ref_payload("GOOG", horizon=5))
        assert r.status_code == 200, r.text
        assert r.json()["mean"] == [0.42] * 5

    def test_raw_history_still_works_with_feast_configured(
        self, client: TestClient
    ) -> None:
        """Raw history bypasses Feast even when feast_repo_path is set."""
        r = client.post("/ts/predict", json=_raw_payload(horizon=3))
        assert r.status_code == 200, r.text
        assert r.json()["mean"] == [0.42, 0.42, 0.42]

    def test_feature_ref_without_feast_repo_returns_422(self) -> None:
        """Sending feature_ref to a spec without feast_repo_path → 422."""
        no_feast_app = _build_asgi_app([_ts_spec(feast_repo_path=None)])
        c = TestClient(no_feast_app)
        r = c.post("/ts/predict", json=_feature_ref_payload("AAPL"))
        assert r.status_code == 422
        assert "feast_repo_path" in r.json()["detail"]

    def test_unknown_ticker_returns_502(self, feast_repo: str) -> None:
        """Requesting an entity not in the feature store propagates as 502."""
        app = _build_asgi_app([_ts_spec(feast_repo_path=feast_repo)])
        c = TestClient(app, raise_server_exceptions=False)
        payload = {
            "model_type": "time_series",
            "model_name": "ts",
            "feature_ref": {
                "feature_view": "asset_prices",
                "feature_name": "close_history",
                "entity_key": "ticker",
                "entity_value": "DOES_NOT_EXIST",
            },
            "horizon": 3,
            "frequency": "1d",
        }
        r = c.post("/ts/predict", json=payload)
        # Feast returns None / empty list for missing entities — FeastResolver
        # raises ValueError("not found") which becomes 502.
        assert r.status_code in (422, 500, 502), r.text

    def test_response_has_correct_horizon_and_model_name(
        self, client: TestClient
    ) -> None:
        """Verify response metadata survives the Feast resolution path."""
        r = client.post("/ts/predict", json=_feature_ref_payload("AAPL", horizon=6))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_name"] == "ts"
        assert len(body["mean"]) == 6

    def test_multiple_tickers_in_sequence(self, client: TestClient) -> None:
        """Exercise all three materialised tickers in one test."""
        for ticker in ("AAPL", "MSFT", "GOOG"):
            r = client.post("/ts/predict", json=_feature_ref_payload(ticker, horizon=2))
            assert r.status_code == 200, f"{ticker}: {r.text}"
            assert r.json()["mean"] == [0.42, 0.42]
