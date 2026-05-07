# Contributing to Sheaf

Thanks for your interest. Sheaf is early — the best contributions right now are new backend implementations and feedback on the API contracts.

## Development setup

```bash
git clone https://github.com/korbonits/sheaf.git
cd sheaf
uv sync --extra dev --extra batch --extra worker   # mirrors CI; required for full test pass
uv run pre-commit install                          # wire up pre-commit hooks (required, per-clone)
uv run pytest tests/
```

The `--extra batch` and `--extra worker` flags pull in `pandas`/`pyarrow` (Ray Data) and `fakeredis` respectively — without them, `tests/test_batch.py` skips and `tests/test_smoke_worker.py` is a no-op.  CI runs with the same combination.

## What to contribute

### New backends (highest value)

Each new model type needs:
1. A typed request/response contract in `src/sheaf/api/`
2. A `ModelBackend` implementation in `src/sheaf/backends/`
3. Tests in `tests/`

The time series contract (`src/sheaf/api/time_series.py`) and Chronos2 backend (`src/sheaf/backends/chronos.py`) are the reference implementation. Follow that pattern.

**What's wanted now**: the v0.5 wishlist (LiDAR, pose, optical flow, multimodal generation, controllable TTS) all shipped in v0.5.1; v0.6 added offline `BatchRunner`; v0.7 added the `SheafWorker` async-job queue; v0.8 added LoRA adapter multiplexing for FLUX + SDXL.  Open an issue if you have a model type in mind that isn't in the table below — the contract design is the interesting part, not the integration plumbing.

### API contract feedback

If you have strong opinions about what a request/response contract should look like for a new modality — open an issue. Getting the contracts right before building the optimizations behind them is the priority at this stage.

### Bug reports

Please include:
- Python version
- `sheaf-serve` version (`pip show sheaf-serve`)
- Minimal reproducer

## Adding a backend

Register your backend with the `@register_backend` decorator:

```python
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

@register_backend("my-model")
class MyModelBackend(ModelBackend):
    def __init__(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id
        self._model = None

    @property
    def model_type(self) -> str:
        return "time_series"  # or whichever ModelType applies

    def load(self) -> None:
        # Load weights here — called once at server startup
        self._model = load_my_model(self._model_id)

    def predict(self, request) -> BaseResponse:
        # Single request inference
        ...

    def batch_predict(self, requests) -> list[BaseResponse]:
        # Override for model-type-aware batching
        # Default: runs predict() sequentially
        ...
```

Then add your model's optional dependencies to `pyproject.toml` under `[project.optional-dependencies]`.

## Implemented backends

All backends below are implemented, tested, and wired into the Ray Serve smoke suite:

| Backend | Registry key | Install extra | Notes |
|---|---|---|---|
| Chronos2 | `chronos2` | `time-series` | Chronos-Bolt and Chronos-T5 families |
| TimesFM | `timesfm` | `time-series` | Google TimesFM |
| Moirai | `moirai` | `moirai` | Salesforce Moirai (uni2ts) |
| TabPFN | `tabpfn` | `tabular` | Classification + regression; requires `TABPFN_TOKEN` |
| Whisper | `whisper` | `audio` | openai-whisper (PyTorch) |
| FasterWhisper | `faster_whisper` | `audio` | CTranslate2; no torch at inference |
| MusicGen | `musicgen` | `audio-generation` | Meta MusicGen; text → audio |
| Bark | `bark` | `tts` | Suno Bark via transformers |
| Kokoro | `kokoro` | `kokoro` | Voice + speed per request; reuses `TTSRequest` |
| OpenCLIP | `open_clip` | `vision` | Image and text embeddings |
| DINOv2 | `dinov2` | `vision` | Image embeddings; CLS or mean pooling |
| SAM2 | `sam2` | `vision` | Prompted segmentation |
| DepthAnything | `depth_anything` | `vision` | Monocular depth estimation |
| DETR | `detr` | `vision` | Object detection; any `AutoModelForObjectDetection` |
| ViTPose | `vitpose` | `pose` | Top-down COCO 17-keypoint estimation |
| RAFT | `raft` | `optical-flow` | torchvision raft_large/raft_small |
| ESM-3 | `esm3` | `molecular` | Protein embeddings; Python 3.12+ |
| NucleotideTransformer | `nucleotide_transformer` | `genomics` | DNA/RNA embeddings |
| MolFormer | `molformer` | `small-molecule` | Small molecule SMILES embeddings |
| MACE | `mace` | `materials` | Universal interatomic potential (energy, forces, stress) |
| Prithvi | `prithvi` | `earth-observation` | IBM/NASA geospatial embeddings |
| GraphCast | `graphcast` | `weather` | Google DeepMind weather forecasting |
| ImageBind | `imagebind` | `multimodal` | Cross-modal embeddings; install from source (not on PyPI) |
| FLUX | `flux` | `diffusion` | FLUX.1-schnell / FLUX.1-dev image generation |
| SDXL | `sdxl` | `multimodal-generation` | img2img + inpainting; mode set at `__init__` |
| VideoMAE | `videomae` | `video` | VideoMAE / TimeSformer embeddings + Kinetics-400 classification |
| PointNet | `pointnet` | `lidar` | Pure-PyTorch 3D point cloud embed + ModelNet40 classify |

## Feast feature store integration

`TimeSeriesRequest` accepts either raw `history` or a `feature_ref` — a pointer into a Feast online feature store. The serving layer resolves the feature before inference; backends always see `history`.

To use Feast, set `feast_repo_path` on `ModelSpec`:

```python
spec = ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    feast_repo_path="/feast/feature_repo",
)
```

Then send requests with `feature_ref` instead of `history`:

```json
{
    "model_type": "time_series",
    "model_name": "chronos",
    "feature_ref": {
        "feature_view": "asset_prices",
        "feature_name": "close_history_30d",
        "entity_key": "ticker",
        "entity_value": "AAPL"
    },
    "horizon": 7,
    "frequency": "1d"
}
```

See `examples/quickstart_feast.py` for a full end-to-end example with a local SQLite store.

## Modal serverless deployment

`ModalServer` is a zero-infra alternative to `ModelServer` (Ray Serve). It wraps the same `ModelSpec` list and deploys to Modal's managed GPU infrastructure:

```python
from sheaf import ModalServer

server = ModalServer(models=[spec], app_name="my-sheaf", gpu="A10G")
app = server.app  # modal deploy my_server.py
```

See `examples/quickstart_modal.py` for a full example.

## Offline batch inference

`BatchRunner` runs any registered backend over a JSONL source → JSONL sink via Ray Data `map_batches`.  Same backend, same typed contract — the request/response classes are shared with the Ray Serve and Modal paths.

```python
from sheaf.batch import BatchRunner, BatchSpec, JsonlSource, JsonlSink

spec = BatchSpec(
    name="chronos-batch",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={"model_id": "amazon/chronos-bolt-tiny", "device_map": "cpu"},
    source=JsonlSource(path="in.jsonl"),
    sink=JsonlSink(path="out.jsonl"),
    batch_size=64,
)
BatchRunner(spec).run()
```

Two execution modes:

- **`compute="tasks"`** (default): stateless tasks with a worker-local backend cache.  `load()` fires once per worker process.  Best for cheap loads.
- **`compute="actors"`**: actor pool of size `num_actors`; `load()` fires once per actor and the loaded model persists for the actor's lifetime.  Best for FLUX / GraphCast / SDXL.

See `examples/quickstart_batch.py` and `examples/quickstart_batch_actors.py`.  Install with `pip install 'sheaf-serve[batch]'`.

## Async-job worker

`SheafWorker` consumes jobs from a queue (Redis Streams in v1; pluggable `JobQueue` / `ResultStore` ABCs for SQS / Kafka / Postgres later).  For inference where HTTP request/response is the wrong shape — FLUX 50-step, GraphCast multi-day rollouts, large-batch SDXL.

```python
from sheaf.api.base import ModelType
from sheaf.worker import (
    SheafWorker, WorkerSpec, RedisStreamsQueue, RedisHashResultStore,
)

spec = WorkerSpec(
    name="flux-worker",
    model_type=ModelType.DIFFUSION,
    backend="flux",
    queue=RedisStreamsQueue(stream="sheaf:flux", group="workers", consumer="w1"),
    results=RedisHashResultStore(prefix="sheaf:flux:result"),
    max_retries=3,
)
SheafWorker(spec).start()
```

Clients enqueue via `JobQueueClient.enqueue(request, webhook_url=None)` and either poll `wait_for_result(job_id)` or wait for a webhook POST on completion.  At-least-once delivery (XACK only after persist); jobs that exceed `max_retries` go to a dead-letter stream and a `status="failed"` `JobResult` is written so polling clients don't hang.

See `examples/quickstart_worker.py`.  Install with `pip install 'sheaf-serve[worker]'`.

## LoRA adapter multiplexing

Diffusion backends (`flux`, `sdxl`) opt in to LoRA via `supports_lora()` returning `True`.  Declare the adapter registry on the spec, and select per request:

```python
from sheaf.lora import LoRAAdapter, LoRAConfig

spec = ModelSpec(
    name="flux-loras",
    model_type=ModelType.DIFFUSION,
    backend="flux",
    backend_kwargs={"model_id": "black-forest-labs/FLUX.1-schnell"},
    resources=ResourceConfig(num_gpus=1),
    lora=LoRAConfig(
        adapters={
            "sketch":     LoRAAdapter(source="/loras/sketch.safetensors", weight=0.8),
            "watercolor": LoRAAdapter(source="hf:user/watercolor-lora",    weight=1.0),
        },
        default="sketch",
    ),
)
```

Requests pick adapters via `DiffusionRequest.adapters` (or `MultimodalGenerationRequest.adapters` for SDXL) and optionally override per-request weights via `adapter_weights`.  When `spec.lora` is set, `_SheafDeployment` automatically buckets requests by their resolved `(names, weights)` selection inside each Ray Serve batch window — `pipeline.set_adapters` is process-global state, so concurrent different-adapter requests must dispatch separately.

To add LoRA support to a new backend, override `supports_lora()`, `load_adapters(adapters)`, and `set_active_adapters(names, weights)` on `ModelBackend`.  Empty `names` should disable LoRAs cleanly (e.g. `pipeline.disable_lora()` for diffusers backends — `set_adapters([], [])` raises `KeyError`).  See `src/sheaf/backends/flux.py` and `src/sheaf/backends/sdxl.py` for the reference implementation, and `examples/quickstart_flux_lora.py` (Ray Serve) / `examples/quickstart_flux_lora_modal.py` (Modal A100) for end-to-end deploy examples.

## Code style

```bash
uv run ruff check src/ tests/         # lint (check)
uv run ruff check --fix src/ tests/   # lint (auto-fix)
uv run ruff format src/ tests/        # format
uv run ty check src/                  # type check
```

Pre-commit hooks run the same checks as CI (`ruff check`, `ruff format --check`, `ty check`) and block the commit if anything fails. Run `uv run pre-commit install` once after cloning to activate them.

## License

By contributing, you agree your contributions will be licensed under Apache 2.0.
