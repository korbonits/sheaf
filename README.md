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
pip install sheaf-serve                        # core only
pip install "sheaf-serve[time-series]"         # + Chronos2 / TimesFM
pip install "sheaf-serve[tabular]"             # + TabPFN
pip install "sheaf-serve[molecular]"           # + ESM-3
pip install "sheaf-serve[all]"                 # everything
```

## Quickstart

```python
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.scheduling import BatchPolicy
from sheaf.spec import ResourceConfig

chronos_spec = ModelSpec(
    name="chronos2-small",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={"model_id": "amazon/chronos-bolt-small"},
    resources=ResourceConfig(num_gpus=1, replicas=2),
    batch_policy=BatchPolicy(max_batch_size=64, bucket_by="horizon"),
)

server = ModelServer(models=[chronos_spec])
server.run()
```

Then send a forecast request:

```python
import requests
from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest

req = TimeSeriesRequest(
    model_name="chronos2-small",
    history=[1.2, 1.5, 1.3, 1.8, 2.1, 2.4, 2.2, 2.7],
    horizon=24,
    frequency=Frequency.HOURLY,
    output_mode=OutputMode.QUANTILES,
)

response = requests.post(
    "http://localhost:8000/chronos2-small",
    json=req.model_dump(mode="json"),
)
print(response.json())
# {"mean": [...], "quantiles": {"0.1": [...], "0.5": [...], "0.9": [...]}, ...}
```

Or resolve features directly from Feast:

```python
req = TimeSeriesRequest(
    model_name="chronos2-small",
    feature_ref={"feature_view": "asset_prices", "entity_id": "AAPL"},
    horizon=24,
    frequency=Frequency.HOURLY,
)
```

---

## Supported model types

| Type | Status | Backends |
|---|---|---|
| Time series | ✅ v0.1 | Chronos2, TimesFM |
| Tabular | 🔜 v0.2 | TabPFN |
| Molecular / biological | 🔜 v0.2 | ESM-3, AlphaFold |
| Audio | 🔜 v0.3 | Whisper, MusicGen |
| Embeddings | 🔜 v0.3 | CLIP, ColBERT |
| Geospatial / Earth science | 🔜 v0.3 | GraphCast, Clay |
| Diffusion | 🔜 v0.4 | Flux, Stable Diffusion |
| Neural operators | 🔜 v0.4 | FNO, DeepONet |

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
