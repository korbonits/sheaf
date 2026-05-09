# Client SDK

The typed Python client lives in the same package as the server —
`pip install sheaf-serve` gets you both. No separate `sheaf-client`
package; one source of truth for the request/response schemas.

```python
from sheaf.client import SheafClient

client = SheafClient(base_url="http://127.0.0.1:8000")
```

Async variant: `AsyncSheafClient` with the same surface but `async def`
methods. Streaming is async-only; everything else is available in both
forms.

## Predict

`client.predict(deployment, request)` decodes the response into a typed
Pydantic object — `TimeSeriesResponse`, `EmbeddingResponse`,
`DiffusionResponse`, etc.

```python
from sheaf.api.time_series import TimeSeriesRequest, Frequency, OutputMode
from sheaf.client import SheafClient

client = SheafClient(base_url="http://127.0.0.1:8000")

response = client.predict(
    "chronos",
    TimeSeriesRequest(
        model_name="chronos",
        history=[100, 110, 105, 120, 125, 130, 135],
        horizon=6,
        frequency=Frequency.HOURLY,
        output_mode=OutputMode.QUANTILES,
        quantile_levels=[0.1, 0.5, 0.9],
    ),
)
print(response.mean)
print(response.quantiles["0.5"])
```

Validation runs before the wire call — a malformed request raises
`pydantic.ValidationError` locally, not after a server round-trip.

## Health + readiness

```python
client.health("chronos")  # liveness — Ray Serve replica is up
client.ready("chronos")   # readiness — backend.load() finished
```

Both raise on non-200; the underlying `SheafError` subclass tells you
whether it was a 4xx (`ClientError`) or 5xx (`ServerError`).

## Streaming (async only)

```python
import asyncio
from sheaf.api.diffusion import DiffusionRequest
from sheaf.client import AsyncSheafClient

async def main():
    async with AsyncSheafClient(base_url="https://flux.example.com") as client:
        async for event in client.stream(
            "flux",
            DiffusionRequest(
                model_name="flux",
                prompt="a sheaf, mathematical concept, line drawing",
                num_inference_steps=20,
            ),
        ):
            if event["type"] == "progress":
                print(f"step {event['step']}/{event['total_steps']}")
            elif event["type"] == "result":
                save_image(event["image_b64"])
                break

asyncio.run(main())
```

## Retry config

```python
from sheaf.client import RetryConfig, SheafClient

client = SheafClient(
    base_url="http://127.0.0.1:8000",
    retry=RetryConfig(
        max_attempts=3,
        backoff_factor=0.5,         # exponential: 0.5s, 1s, 2s
        retry_on_status=(502, 503, 504),
        retry_on_connection_errors=True,
    ),
)
```

`RetryConfig` is a frozen dataclass — pass it once at client
construction. Streaming bypasses retry by design (replaying a stream
mid-flight is more often wrong than right).

## request_id propagation

Every `BaseRequest` carries a `request_id` (UUID, auto-generated if
not set). The server logs and metrics include it; the client
decorates every raised `SheafError` with it.

```python
try:
    response = client.predict("chronos", req)
except SheafError as e:
    print(f"server error for {e.request_id}: {e}")
    # → "server error for 7d4e8a... — 503 Service Unavailable"
```

Use it to correlate a client-side failure to a server log line
without grepping by timestamp.

## Reference

Full schema in the [Client API reference](../api-reference/client.md).
