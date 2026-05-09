# Quickstart

This page goes from `pip install` to a running HTTP server with a real
forecast in under five minutes. No GPU required; the example model is
`amazon/chronos-bolt-tiny` (~80 MB, CPU-friendly).

## 1. Install

```bash
pip install "sheaf-serve[time-series]"
```

The base `sheaf-serve` package gives you the orchestrator and contracts.
Each model family lives behind an extras flag — here `[time-series]`
brings in [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting).
The full matrix is on the [Models](models/index.md) page.

## 2. Direct backend (no server, no Ray)

The simplest possible call — no HTTP, no Ray, no batching. Useful for
notebooks and unit tests.

```python
from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest
from sheaf.backends.chronos import Chronos2Backend

backend = Chronos2Backend(
    model_id="amazon/chronos-bolt-tiny",
    device_map="cpu",
    torch_dtype="float32",
)
backend.load()

req = TimeSeriesRequest(
    model_name="chronos-bolt-tiny",
    history=[312, 298, 275, 260, 255, 263, 285, 320,
             368, 402, 421, 435, 442, 438, 430, 425],
    horizon=12,
    frequency=Frequency.HOURLY,
    output_mode=OutputMode.QUANTILES,
    quantile_levels=[0.1, 0.5, 0.9],
)

response = backend.predict(req)
print(response.mean)            # list[float]
print(response.quantiles["0.5"])  # median forecast
```

Already useful, but you're loading the model on every process. The next
step is to put it behind a server.

## 3. ModelServer (Ray Serve)

```python
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.spec import ResourceConfig

server = ModelServer(models=[
    ModelSpec(
        name="chronos",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        backend_kwargs={
            "model_id": "amazon/chronos-bolt-tiny",
            "device_map": "cpu",
            "torch_dtype": "float32",
        },
        resources=ResourceConfig(num_cpus=1, replicas=1),
    ),
])
server.run()  # blocks until the deployment is ready
```

The deployment is now available on `http://127.0.0.1:8000` with three
routes mounted under `/chronos/`:

| Route | Method | Purpose |
|---|---|---|
| `/chronos/predict` | POST | typed forecast call |
| `/chronos/health` | GET | liveness — Ray Serve replica is up |
| `/chronos/ready` | GET | readiness — backend.load() finished |
| `/chronos/stream` | POST | SSE streaming variant (per-deployment) |
| `/metrics` | GET | Prometheus metrics |

Hit it from anywhere — curl works:

```bash
curl -X POST http://127.0.0.1:8000/chronos/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "time_series",
    "model_name": "chronos",
    "history": [312, 298, 275, 260, 255, 263, 285, 320],
    "horizon": 6,
    "frequency": "1h",
    "output_mode": "mean"
  }'
```

## 4. From Python: the typed client

```bash
pip install sheaf-serve  # the client lives in the same package
```

```python
from sheaf.client import SheafClient
from sheaf.api.time_series import TimeSeriesRequest, Frequency, OutputMode

client = SheafClient(base_url="http://127.0.0.1:8000")

response = client.predict(
    "chronos",
    TimeSeriesRequest(
        model_name="chronos",
        history=[312, 298, 275, 260, 255, 263, 285, 320],
        horizon=6,
        frequency=Frequency.HOURLY,
        output_mode=OutputMode.MEAN,
    ),
)
print(response.mean)
```

`SheafClient.predict` returns a typed response (e.g. `TimeSeriesResponse`)
decoded from the server's JSON. Async variant: `AsyncSheafClient` with
the same API but `async def`. See [Client](client/index.md).

## 5. Add a second model

The whole point of sheaf-serve is heterogeneous serving — multiple model
types, different contracts, the same Ray cluster. Add a vision embedding
deployment alongside the forecaster:

```python
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.spec import ResourceConfig

server = ModelServer(models=[
    ModelSpec(
        name="forecaster",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        backend_kwargs={"model_id": "amazon/chronos-bolt-tiny"},
    ),
    ModelSpec(
        name="embedder",
        model_type=ModelType.EMBEDDING,
        backend="openclip",
        backend_kwargs={"model_name": "ViT-B-32"},
        resources=ResourceConfig(num_gpus=0.25),
    ),
])
server.run()
```

Now `POST /forecaster/predict` takes a `TimeSeriesRequest` and `POST
/embedder/predict` takes an `EmbeddingRequest`. Each deployment has its
own contract, batching, and cache.

## 6. Where to go from here

- **Production deployment** — bake into a Docker image:
  [Deployment / Docker](deployment/docker.md). For Kubernetes, use the
  [KubeRay](deployment/kuberay.md) example.
- **Heavy I/O job** — replace the live HTTP path with the
  [offline batch runner](concepts/batch.md): JSONL in, JSONL out, via
  Ray Data.
- **Async fan-out** — for jobs that take longer than a request can wait,
  use the [`SheafWorker`](concepts/worker.md) — Redis Streams in,
  webhook out.
- **Optimise the steady state** — turn on
  [request caching](concepts/caching.md), tune
  [batching](concepts/batching.md), and bucket length-variable inputs
  with `bucket_by`.
- **Streaming output** — for FLUX progress events or chunked transcripts,
  see [Streaming](concepts/streaming.md).
