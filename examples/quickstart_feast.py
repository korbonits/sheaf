"""Sheaf + Feast feature store quickstart.

This example shows how to use a Feast online feature store as the input source
for time series predictions instead of passing raw history in the request.

The flow:
  1. Build a local Feast repo (SQLite online store — no external services).
  2. Define an "asset_prices" feature view that stores 30-day close history
     as a list[float] for each ticker.
  3. Materialise a few rows into the online store.
  4. Start a Sheaf ASGI server with feast_repo_path wired to the repo.
  5. Send requests with feature_ref (no raw history) and print predictions.

Install:
    pip install 'sheaf-serve[time-series,feast]'

Run:
    python examples/quickstart_feast.py
"""

from __future__ import annotations

import datetime
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Check deps before the heavy imports
# ---------------------------------------------------------------------------

try:
    import feast  # noqa: F401
except ImportError:
    print("feast is required. Install with: pip install 'sheaf-serve[feast]'")
    sys.exit(1)

try:
    import chronos  # noqa: F401
except ImportError:
    print(
        "chronos-forecasting is required for the time-series backend.\n"
        "Install with: pip install 'sheaf-serve[time-series]'"
    )
    sys.exit(1)

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Array, Float32
from starlette.testclient import TestClient

from sheaf.api.base import ModelType
from sheaf.modal_server import _build_asgi_app
from sheaf.spec import ModelSpec, ResourceConfig

# ---------------------------------------------------------------------------
# Sample data — 30-day closing prices for three tickers (made-up values)
# ---------------------------------------------------------------------------

_HISTORY: dict[str, list[float]] = {
    "AAPL": [
        170.1,
        171.5,
        172.0,
        170.8,
        173.2,
        174.5,
        175.0,
        173.8,
        176.0,
        177.2,
        178.5,
        177.0,
        179.3,
        180.1,
        179.8,
        181.0,
        182.5,
        181.3,
        183.0,
        184.2,
        183.5,
        185.0,
        186.3,
        185.8,
        187.0,
        188.5,
        187.2,
        189.0,
        190.3,
        191.0,
    ],
    "MSFT": [
        330.0,
        332.5,
        331.0,
        334.0,
        336.3,
        335.1,
        338.0,
        337.5,
        340.0,
        341.2,
        343.5,
        342.0,
        344.3,
        345.1,
        344.8,
        346.0,
        347.5,
        346.3,
        348.0,
        349.2,
        348.5,
        350.0,
        351.3,
        350.8,
        352.0,
        353.5,
        352.2,
        354.0,
        355.3,
        356.0,
    ],
    "GOOG": [
        140.0,
        141.0,
        143.0,
        142.5,
        144.0,
        143.5,
        145.0,
        144.2,
        146.0,
        147.3,
        148.5,
        147.0,
        149.3,
        150.1,
        149.8,
        151.0,
        152.5,
        151.3,
        153.0,
        154.2,
        153.5,
        155.0,
        156.3,
        155.8,
        157.0,
        158.5,
        157.2,
        159.0,
        160.3,
        161.0,
    ],
}

# ---------------------------------------------------------------------------
# Step 1 — Build the Feast repo in a temp directory
# ---------------------------------------------------------------------------


def build_feast_repo(repo_dir: str) -> None:
    print(f"Building Feast repo at {repo_dir} ...")

    # feature_store.yaml — local provider, SQLite online store
    yaml = """\
project: sheaf_quickstart
registry: registry.db
provider: local
online_store:
    type: sqlite
    path: online_store.db
entity_key_serialization_version: 2
"""
    with open(os.path.join(repo_dir, "feature_store.yaml"), "w") as fh:
        fh.write(yaml)

    # DataFrame with one row per ticker
    now = datetime.datetime.utcnow()
    df = pd.DataFrame(
        {
            "ticker": list(_HISTORY.keys()),
            "close_history_30d": list(_HISTORY.values()),
            "event_timestamp": [now] * len(_HISTORY),
            "created": [now] * len(_HISTORY),
        }
    )
    parquet_path = os.path.join(repo_dir, "prices.parquet")
    df.to_parquet(parquet_path, index=False)

    # Entity + FeatureView definitions
    ticker_entity = Entity(name="ticker", join_keys=["ticker"])
    source = FileSource(
        path=parquet_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )
    prices_view = FeatureView(
        name="asset_prices",
        entities=[ticker_entity],
        schema=[Field(name="close_history_30d", dtype=Array(Float32))],
        source=source,
        ttl=datetime.timedelta(days=1),
    )

    store = FeatureStore(repo_path=repo_dir)
    store.apply([ticker_entity, prices_view])

    # Materialise into the online store
    print("Materialising features into SQLite online store ...")
    store.materialize_incremental(
        end_date=datetime.datetime.utcnow() + datetime.timedelta(minutes=1)
    )
    print("Done.\n")


# ---------------------------------------------------------------------------
# Step 2 — Verify the feature store resolves correctly before serving
# ---------------------------------------------------------------------------


def verify_feast_resolution(repo_dir: str) -> None:
    from sheaf.integrations.feast import FeastResolver

    resolver = FeastResolver(repo_dir)
    resolver.load()

    print("Verifying Feast resolution:")
    for ticker in _HISTORY:
        history = resolver.resolve(
            __import__("sheaf.api.time_series", fromlist=["FeatureRef"]).FeatureRef(
                feature_view="asset_prices",
                feature_name="close_history_30d",
                entity_key="ticker",
                entity_value=ticker,
            )
        )
        n = len(history)
        assert n == 30, f"Expected 30 values for {ticker}, got {n}"
        print(
            f"  {ticker}: [{history[0]:.1f}, {history[1]:.1f}, ..., {history[-1]:.1f}]"
            f"  ({n} values)"
        )
    print()


# ---------------------------------------------------------------------------
# Step 3 — Start Sheaf and send feature_ref requests
# ---------------------------------------------------------------------------


def run_predictions(repo_dir: str) -> None:
    # For the quickstart we use Chronos-Bolt-Small (downloads ~300 MB on first run)
    spec = ModelSpec(
        name="chronos-small",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        backend_kwargs={"model_id": "amazon/chronos-bolt-small", "device_map": "cpu"},
        resources=ResourceConfig(num_cpus=1),
        feast_repo_path=repo_dir,
    )

    print("Loading Chronos backend (this downloads weights on first run) ...")
    app = _build_asgi_app([spec])
    client = TestClient(app)
    print("Server ready.\n")

    print("Sending feature_ref requests (no raw history in the payload):")
    print("-" * 60)

    for ticker in _HISTORY:
        payload = {
            "model_type": "time_series",
            "model_name": "chronos-small",
            "feature_ref": {
                "feature_view": "asset_prices",
                "feature_name": "close_history_30d",
                "entity_key": "ticker",
                "entity_value": ticker,
            },
            "horizon": 7,
            "frequency": "1d",
            "output_mode": "quantiles",
            "quantile_levels": [0.1, 0.5, 0.9],
        }

        r = client.post("/chronos-small/predict", json=payload)
        if r.status_code != 200:
            print(f"{ticker}: ERROR {r.status_code} — {r.text}")
            continue

        body = r.json()
        # quantiles: dict[str, list[float]] — keys "0.1", "0.5", "0.9"
        q = body["quantiles"]
        p10, p50, p90 = q["0.1"], q["0.5"], q["0.9"]
        last_price = _HISTORY[ticker][-1]

        print(f"\n{ticker}  (last close: ${last_price:.2f})")
        print("  7-day forecast (p10 / median / p90):")
        for day, (lo_v, mid_v, hi_v) in enumerate(zip(p10, p50, p90), 1):
            print(f"    day {day}: ${lo_v:7.2f} / ${mid_v:7.2f} / ${hi_v:7.2f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    with tempfile.TemporaryDirectory() as repo_dir:
        build_feast_repo(repo_dir)
        verify_feast_resolution(repo_dir)
        run_predictions(repo_dir)

    print("Quickstart complete.")


if __name__ == "__main__":
    main()
