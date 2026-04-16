# Sheaf

[![PyPI](https://img.shields.io/pypi/v/sheaf-serve)](https://pypi.org/project/sheaf-serve/)
[![CI](https://github.com/korbonits/sheaf/actions/workflows/ci.yml/badge.svg)](https://github.com/korbonits/sheaf/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/sheaf-serve)](https://pypi.org/project/sheaf-serve/)

**Unified serving layer for non-text foundation models.**

vLLM solved inference for text LLMs by defining a standard compute contract and optimizing behind it. The same problem exists for every other class of foundation model — time series, tabular, molecular, geospatial, diffusion, audio — and nobody has solved it. Sheaf is that solution.

Each model type gets a typed request/response contract. Batching, caching, and scheduling are optimized per model type. [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) is the substrate. [Feast](https://feast.dev) is a first-class input primitive.

> *In mathematics, a sheaf tracks locally-defined data that glues consistently across a space. Each model type defines its own local contract; Sheaf ensures they cohere into a unified serving layer.*

---

## Install

```bash
pip install sheaf-serve                           # core only
pip install "sheaf-serve[time-series]"            # + Chronos2 / TimesFM / Moirai
pip install "sheaf-serve[tabular]"                # + TabPFN
pip install "sheaf-serve[molecular]"              # + ESM-3  (Python 3.12+)
pip install "sheaf-serve[genomics]"               # + Nucleotide Transformer
pip install "sheaf-serve[small-molecule]"         # + MolFormer
pip install "sheaf-serve[materials]"              # + MACE-MP
pip install "sheaf-serve[audio]"                  # + Whisper / faster-whisper
pip install "sheaf-serve[audio-generation]"       # + MusicGen
pip install "sheaf-serve[tts]"                    # + Bark
pip install "sheaf-serve[vision]"                 # + DINOv2 / OpenCLIP / SAM2 / Depth Anything / DETR
pip install "sheaf-serve[earth-observation]"      # + Prithvi
pip install "sheaf-serve[weather]"                # + GraphCast
pip install "sheaf-serve[feast]"                  # + Feast feature store integration
pip install "sheaf-serve[modal]"                  # + Modal serverless deployment
pip install "sheaf-serve[all]"                    # everything
```

## Quickstart

**Direct backend inference:**

```python
from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest
from sheaf.backends.chronos import Chronos2Backend

backend = Chronos2Backend(model_id="amazon/chronos-bolt-tiny", device_map="cpu")
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
# response.mean, response.quantiles
```

**Ray Serve (production, autoscaling):**

```python
from sheaf import ModelServer
from sheaf.spec import ModelSpec, ResourceConfig
from sheaf.api.base import ModelType

server = ModelServer(models=[
    ModelSpec(
        name="chronos",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        backend_kwargs={"model_id": "amazon/chronos-bolt-small"},
        resources=ResourceConfig(num_gpus=1),
    ),
])
server.run()  # POST /chronos/predict, GET /chronos/health
```

**Feast feature store (resolve features at request time):**

```python
# ModelSpec wires Feast — no history needed in the request
spec = ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    feast_repo_path="/feast/feature_repo",
)

# Client sends feature_ref instead of raw history
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

**Modal (serverless, zero-infra):**

```python
from sheaf import ModalServer

server = ModalServer(models=[spec], app_name="my-sheaf", gpu="A10G")
app = server.app  # modal deploy my_server.py
```

See [`examples/`](examples/) for time series comparison, tabular, audio, vision, and the Feast feature store quickstart.

---

## Supported model types

| Type | Status | Backends |
|---|---|---|
| Time series | ✅ v0.1 | Chronos2, Chronos-Bolt, TimesFM, Moirai |
| Tabular | ✅ v0.1 | TabPFN v2 |
| Audio transcription | ✅ v0.3 | Whisper, faster-whisper |
| Audio generation | ✅ v0.3 | MusicGen |
| Text-to-speech | ✅ v0.3 | Bark |
| Vision embeddings | ✅ v0.3 | OpenCLIP, DINOv2 |
| Segmentation | ✅ v0.3 | SAM2 |
| Depth estimation | ✅ v0.3 | Depth Anything v2 |
| Object detection | ✅ v0.3 | DETR / RT-DETR |
| Protein / molecular | ✅ v0.3 | ESM-3 (Python 3.12+) |
| Genomics | ✅ v0.3 | Nucleotide Transformer |
| Small molecule | ✅ v0.3 | MolFormer-XL |
| Materials science | ✅ v0.3 | MACE-MP-0 |
| Earth observation | ✅ v0.3 | Prithvi (IBM/NASA) |
| Weather forecasting | ✅ v0.3 | GraphCast |
| Cross-modal embeddings | ✅ v0.3 | ImageBind (text, vision, audio, depth, thermal) |
| Feast feature store | ✅ v0.3 | Any Feast online store (SQLite, Redis, DynamoDB, …) |
| Modal serverless | ✅ v0.3 | `ModalServer` — zero-infra GPU deployment |
| Diffusion / image gen | 🔜 v0.4 | FLUX |
| Neural operators | 🔜 v0.4 | FNO, DeepONet |
| Video understanding | 🔜 v0.4 | VideoMAE, TimeSformer |

## Roadmap to production

**v0.2 — serving layer (complete)**
- [x] Ray Serve integration tested end-to-end
- [x] Async `predict()` handlers
- [x] HTTP API with proper request validation (422 on bad input)
- [x] Health check and readiness probe endpoints
- [x] Batching scheduler (BatchPolicy wired into `@serve.batch` per deployment)
- [x] Error handling at the service boundary (backend exceptions → structured HTTP 500)
- [x] Model hot-swap without restart (`ModelServer.update()`)
- [x] Container-friendly auth for TabPFN v2 (`TABPFN_TOKEN` env var)

**v0.3 — model types + integrations (complete)**
- [x] ESM-3 protein embeddings
- [x] Nucleotide Transformer genomics embeddings
- [x] MolFormer-XL small molecule embeddings
- [x] MACE-MP-0 materials (energy, forces, stress)
- [x] Whisper / faster-whisper audio transcription
- [x] MusicGen audio generation
- [x] Bark text-to-speech
- [x] OpenCLIP image/text embeddings
- [x] DINOv2 image embeddings
- [x] SAM2 segmentation
- [x] Depth Anything v2 depth estimation
- [x] DETR / RT-DETR object detection
- [x] Prithvi earth observation embeddings
- [x] GraphCast weather forecasting
- [x] ImageBind cross-modal embeddings (text, vision, audio, depth, thermal)
- [x] Feast feature store integration (`feature_ref` in requests, `FeastResolver`, `feast_repo_path` on `ModelSpec`)
- [x] Modal serverless deployment (`ModalServer` — zero-infra alternative to Ray Serve)

**v0.4 — generation + video**
- [ ] FLUX diffusion / image generation
- [ ] VideoMAE / TimeSformer video understanding

---

## Architecture

```
┌─────────────────────────────────────────┐
│           API Layer                      │  typed contracts per model type
│  TimeSeriesRequest  TabularRequest  ...  │
├─────────────────────────────────────────┤
│         Scheduling Layer                 │  model-type-aware batching
│  BatchPolicy  RequestQueue               │
├─────────────────────────────────────────┤
│          Backend Layer                   │  pluggable execution + Ray Serve
│  ModelBackend  CacheManager  Feast       │
└─────────────────────────────────────────┘
```

**Adding a new backend** takes one class:

```python
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

@register_backend("my-model")
class MyModelBackend(ModelBackend):
    def load(self) -> None:
        self._model = load_my_model()

    def predict(self, request):
        ...

    @property
    def model_type(self):
        return "time_series"
```

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## License

Apache 2.0
