"""SheafClient quickstart — sync + async + streaming against a local ModelServer.

Demonstrates the typed Python client introduced in v0.9.  The same client
talks to both ``ModelServer`` (Ray Serve) and ``ModalServer`` deployments;
the only thing that changes is the ``base_url``.

Prerequisites:
    pip install 'sheaf-serve[time-series]'

    # In another terminal, start a server:
    python examples/quickstart_chronos.py
    # …or any other quickstart that calls server.run().

Then:
    python examples/quickstart_client.py

What this shows:
  - Sync ``SheafClient.predict`` returning a typed ``TimeSeriesResponse``
  - Async ``AsyncSheafClient.predict``
  - ``client.health(...)`` / ``client.ready(...)`` for orchestration
  - Error mapping: 422 → ``ValidationError``, 500 → ``ServerError``
  - SSE streaming via ``client.stream(...)`` (works against any deployment
    whose backend implements ``stream_predict`` — FLUX is the canonical case)
"""

from __future__ import annotations

import asyncio

from sheaf.api.tabular import TabularRequest
from sheaf.api.time_series import Frequency, TimeSeriesRequest
from sheaf.client import (
    AsyncSheafClient,
    SheafClient,
    SheafError,
    ValidationError,
)

BASE_URL = "http://localhost:8000"
DEPLOYMENT = "chronos"  # whatever name your local server registered


# ---------------------------------------------------------------------------
# 1. Sync predict — typed response back
# ---------------------------------------------------------------------------


def sync_predict() -> None:
    print("\n--- sync predict ---")
    with SheafClient(base_url=BASE_URL) as client:
        # health + ready first so we fail fast if nothing's listening
        print("health:", client.health(DEPLOYMENT))
        print("ready :", client.ready(DEPLOYMENT))

        req = TimeSeriesRequest(
            model_name=DEPLOYMENT,
            history=[
                312,
                298,
                275,
                260,
                255,
                263,
                285,
                320,
                368,
                402,
                421,
                435,
                442,
                438,
                430,
                425,
            ],
            horizon=4,
            frequency=Frequency.HOURLY,
        )
        resp = client.predict(DEPLOYMENT, req)
        # resp is a TimeSeriesResponse — full type info, no manual dict access.
        print(f"horizon: {resp.horizon}  mean: {resp.mean[:3]}…")


# ---------------------------------------------------------------------------
# 2. Async predict
# ---------------------------------------------------------------------------


async def async_predict() -> None:
    print("\n--- async predict ---")
    async with AsyncSheafClient(base_url=BASE_URL) as client:
        req = TimeSeriesRequest(
            model_name=DEPLOYMENT,
            history=[1.0, 2.0, 3.0, 4.0, 5.0],
            horizon=3,
            frequency=Frequency.HOURLY,
        )
        resp = await client.predict(DEPLOYMENT, req)
        print(f"async horizon: {resp.horizon}  mean: {resp.mean}")


# ---------------------------------------------------------------------------
# 3. Error mapping — 422 ValidationError, 500 ServerError
# ---------------------------------------------------------------------------


def error_examples() -> None:
    print("\n--- error mapping ---")
    with SheafClient(base_url=BASE_URL) as client:
        # Wrong model_type for the deployment → 422 → ValidationError
        wrong = TabularRequest(
            model_name="t",
            context_X=[[1.0]],
            context_y=[0],
            query_X=[[2.0]],
        )
        try:
            client.predict(DEPLOYMENT, wrong)
        except ValidationError as e:
            print(f"ValidationError ({e.status_code}): {e.detail[:120]}")

        # Unknown deployment → 404 → SheafError (not ValidationError/ServerError)
        try:
            client.predict(
                "does-not-exist",
                TimeSeriesRequest(
                    model_name="x",
                    history=[1.0],
                    horizon=1,
                    frequency=Frequency.HOURLY,
                ),
            )
        except SheafError as e:
            print(f"SheafError ({e.status_code}): {e.detail[:120]}")


# ---------------------------------------------------------------------------
# 4. SSE streaming — works against backends that implement stream_predict
# ---------------------------------------------------------------------------


async def streaming_example() -> None:
    """Replace DEPLOYMENT with a streaming deployment (e.g. FLUX) to try this."""
    print("\n--- streaming (will only produce events on a streaming backend) ---")
    from sheaf.api.diffusion import DiffusionRequest

    async with AsyncSheafClient(base_url=BASE_URL) as client:
        try:
            async for event in client.stream(
                "flux",  # change to a deployment whose backend supports stream_predict
                DiffusionRequest(model_name="flux", prompt="a cat"),
            ):
                print(f"  event: {event['type']}  done={event.get('done')}")
                if event.get("type") == "result":
                    break
        except SheafError as e:
            print(f"  (no streaming deployment available: {e})")


def main() -> None:
    try:
        sync_predict()
    except SheafError as e:
        print(f"could not reach {BASE_URL}: {e}")
        return

    asyncio.run(async_predict())
    error_examples()
    asyncio.run(streaming_example())


if __name__ == "__main__":
    main()
