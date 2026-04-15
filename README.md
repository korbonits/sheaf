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

See [`examples/`](examples/) for time series comparison (Chronos vs TimesFM) and tabular classification/regression.

---

## Supported model types

| Type | Status | Backends |
|---|---|---|
| Time series | ✅ v0.1 | Chronos, Chronos-Bolt, TimesFM |
| Tabular | ✅ v0.1 | TabPFN |
| Molecular / biological | 🔜 v0.3 | ESM-3, AlphaFold |
| Audio | 🔜 v0.3 | Whisper, MusicGen |
| Embeddings | 🔜 v0.3 | CLIP, ColBERT |
| Geospatial / Earth science | 🔜 v0.3 | GraphCast, Clay |
| Diffusion | 🔜 v0.4 | Flux, Stable Diffusion |
| Neural operators | 🔜 v0.4 | FNO, DeepONet |

## Roadmap to production

v0.1 is a library you call from Python. The following are required before Sheaf is production-usable:

**v0.2 — serving layer**
- [ ] Ray Serve integration tested end-to-end
- [ ] Async `predict()` handlers
- [ ] HTTP API with proper request validation (422 on bad input)
- [ ] Health check and readiness probe endpoints
- [ ] Basic error handling at the service boundary

**v0.2 — reliability**
- [ ] Batching scheduler (BatchPolicy currently defined but not enforced)
- [ ] Model hot-swap without restart
- [ ] Container-friendly auth for TabPFN v2 (TABPFN_TOKEN env var works, but first-run browser flow breaks in headless environments)

**v0.3 — model types**
- [ ] ESM-3 / molecular backend
- [ ] Audio backend (Whisper)
- [ ] Geospatial backend (GraphCast)
- [ ] Feast feature resolver implemented end-to-end

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
