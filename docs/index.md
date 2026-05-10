# sheaf-serve

**A unified serving layer for non-text foundation models.**

vLLM solved inference for text LLMs by defining a standard compute contract
and optimising behind it. The same problem exists for every other class of
foundation model — time series, tabular, molecular, geospatial, diffusion,
audio — and nobody has solved it. **sheaf-serve is that solution.**

Each model type gets a typed request/response contract. Batching, caching,
and scheduling are optimised per model type. [Ray Serve](https://docs.ray.io/en/latest/serve/)
is the substrate. [Feast](https://feast.dev) is a first-class input primitive.

> In mathematics, a sheaf tracks locally-defined data that glues consistently
> across a space. Each model type defines its own local contract; sheaf-serve
> ensures they cohere into a unified serving layer.

---

## Install

!!! info "Requires Python 3.11+"
    sheaf-serve targets Python 3.11 and 3.12. macOS users on the
    system `python3` (often 3.10) should bootstrap a 3.11 environment
    first — the easiest way is [`uv`](https://docs.astral.sh/uv/):
    ```bash
    uv venv --python 3.11 .venv && source .venv/bin/activate
    ```
    The `[molecular]` extra (ESM-3) additionally requires Python 3.12+.

```bash
pip install sheaf-serve                       # core
pip install "sheaf-serve[time-series]"        # + Chronos2 / TimesFM / Moirai
pip install "sheaf-serve[vision]"             # + DINOv2 / OpenCLIP / SAM2 / DETR / Depth-Anything
pip install "sheaf-serve[diffusion]"          # + FLUX
pip install "sheaf-serve[multimodal-generation]"  # + SDXL img2img / inpaint
pip install "sheaf-serve[molecular]"          # + ESM-3  (Python 3.12+)
pip install "sheaf-serve[genomics]"           # + Nucleotide Transformer
pip install "sheaf-serve[materials]"          # + MACE-MP
pip install "sheaf-serve[earth-observation]"  # + Prithvi
pip install "sheaf-serve[all]"                # everything
```

The full extras matrix is on the [Models](models/index.md) page.

## Try it without installing — live demo

A public sheaf-serve deployment is running `amazon/chronos-bolt-tiny` on
Modal at:

```
https://korbonits--sheaf-demo-modalserver---init----locals---serve.modal.run
```

Hit it from anywhere. No install, no cluster, no GPU — real time-series
forecasts come back:

```bash
URL=https://korbonits--sheaf-demo-modalserver---init----locals---serve.modal.run

curl $URL/chronos/health
# {"status":"ok"}

curl -X POST $URL/chronos/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "model_type": "time_series",
        "model_name": "chronos",
        "history": [312, 298, 275, 260, 255, 263, 285, 320, 368, 402, 421, 435],
        "horizon": 6,
        "frequency": "1h",
        "output_mode": "quantiles",
        "quantile_levels": [0.1, 0.5, 0.9]
      }'
```

The same `ModelSpec` you'd write locally is what runs there. Source for
the deployment is at
[`examples/demo/app.py`](https://github.com/korbonits/sheaf/blob/main/examples/demo/app.py)
— ~20 lines of user code, deploy with `modal deploy`.

## 30-second taste

```python
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.spec import ResourceConfig

server = ModelServer(models=[
    ModelSpec(
        name="chronos",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        backend_kwargs={"model_id": "amazon/chronos-bolt-small"},
        resources=ResourceConfig(num_gpus=1),
    ),
])
server.run()  # POST /chronos/predict, GET /chronos/health, /metrics, …
```

That's the entire server. Add a second `ModelSpec` to the list and you have
two heterogeneous models on the same Ray cluster, each with their own typed
contract, batching policy, and cache. See [Quickstart](quickstart.md) for
the full walk-through.

## What you get out of the box

<div class="grid cards" markdown>

-   :material-shape:{ .lg .middle } **20+ model types**

    ---

    Time series, tabular, vision, segmentation, depth, detection, pose,
    optical flow, video, audio, TTS, audio generation, image diffusion,
    cross-modal embedding, molecular, genomics, materials, weather,
    earth observation, LiDAR. [Full table →](models/index.md)

-   :material-cog-transfer:{ .lg .middle } **Three execution paths**

    ---

    `ModelServer` (Ray Serve), `ModalServer` (zero-infra serverless),
    `BatchRunner` (offline JSONL → JSONL via Ray Data), plus a
    `SheafWorker` for async-job queues backed by Redis Streams.
    [Deployment →](deployment/index.md)

-   :material-tune:{ .lg .middle } **Per-model-type batching**

    ---

    `@serve.batch` with `bucket_by` for length-bucketing time series by
    horizon or video by frame count. Adapter-aware sub-batching for LoRA.
    [Batching →](concepts/batching.md)

-   :material-database-search:{ .lg .middle } **Feast as a first-class input**

    ---

    Send a `feature_ref` instead of raw history; sheaf-serve resolves
    features from your online store before the model call.
    [Feast →](concepts/feast.md)

-   :material-radio-tower:{ .lg .middle } **SSE streaming**

    ---

    `POST /{name}/stream` for incremental output — FLUX progress events,
    chunked transcription, partial generation. [Streaming →](concepts/streaming.md)

-   :material-package-variant-closed:{ .lg .middle } **Production ready**

    ---

    Structured JSON logging, Prometheus `/metrics`, OTel tracing,
    `Dockerfile` + KubeRay `RayService` example, typed Python client SDK
    with retry + request_id propagation, OpenAPI export.
    [Client →](client/index.md)

</div>

## Why this exists

When you serve a text LLM, you reach for vLLM. When you serve a time-series
foundation model, a tabular foundation model, a protein folding model, a
weather forecasting model — there is no obvious right answer, and so every
team builds the same infrastructure from scratch: a typed request schema,
some batching, a queue, a worker, a Prometheus exporter, a streaming
endpoint, an OpenAPI client, eventually a Dockerfile. That work is
generic across model types.

sheaf-serve is the generic part, factored out, with the typed contracts
shaped per-model-type so users get type safety, not a `dict[str, Any]`
that breaks at request time.

## Where to go next

- New here? Start with the [**Quickstart**](quickstart.md).
- Looking for a specific model type? See the [**Models**](models/index.md) catalog.
- Deploying? Pick a substrate: [Docker](deployment/docker.md),
  [KubeRay](deployment/kuberay.md), or [Modal](deployment/modal.md).
- Calling it from Python? [Client SDK](client/index.md).
