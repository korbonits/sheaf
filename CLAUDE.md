# Sheaf — Claude Code Context

## What this project is

`sheaf-serve` is a unified serving layer for non-text foundation models (time series, tabular, molecular, geospatial, etc.). Think vLLM but for every model type that isn't a text LLM.

Each model type gets a typed request/response contract (Pydantic). Batching, caching, and scheduling are optimized per model type. Ray Serve is the execution substrate. Feast is a first-class input primitive.

PyPI: `pip install sheaf-serve`

## Current state: v0.1.0 (alpha)

v0.1 is a library you call directly from Python — the Ray Serve integration (`ModelServer`) exists in code but is not tested end-to-end. The project is not yet production-usable.

**What works:**
- Time series: Chronos2 and TimesFM backends, full quantile/sample/mean output modes
- Tabular: TabPFN v2 backend, classification + regression

**What doesn't work yet (v0.2 targets):**
- Ray Serve integration tested end-to-end
- Async `predict()` handlers
- HTTP API with proper request validation
- Health check / readiness probe endpoints
- Batching scheduler (BatchPolicy is defined but not enforced)
- Model hot-swap without restart
- Container-friendly auth for TabPFN (TABPFN_TOKEN env var works; first-run browser flow breaks headless)

**v0.3 targets:**
- ESM-3 molecular backend
- Whisper audio backend
- GraphCast geospatial backend
- Feast feature resolver end-to-end

## Repo layout

```
src/sheaf/
  __init__.py          # public exports: ModelServer, ModelSpec
  spec.py              # ModelSpec, ResourceConfig — declares what to serve
  server.py            # ModelServer — Ray Serve orchestrator (stub, not tested e2e)
  registry.py          # @register_backend decorator + _BACKEND_REGISTRY dict
  api/
    base.py            # BaseRequest, BaseResponse, ModelType enum
    time_series.py     # TimeSeriesRequest/Response, Frequency, OutputMode
    tabular.py         # TabularRequest/Response
  backends/
    base.py            # ModelBackend ABC: load(), predict(), batch_predict()
    chronos.py         # Chronos2Backend — Chronos-Bolt + Chronos-T5 families
    tabpfn.py          # TabPFNBackend — TabPFN v2 classification + regression
    timesfm.py         # TimesFMBackend (exists, structure mirrors chronos.py)
  scheduling/
    batch.py           # BatchPolicy (defined, not yet enforced in serving)
  cache/               # stub
  integrations/        # stub
examples/
  quickstart.py        # Chronos time series example
  quickstart_tabular.py
  time_series_comparison.py  # Chronos vs TimesFM
tests/
  test_api.py
  test_tabular_api.py
```

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

## Key design decisions

- **Pydantic for all contracts** — BaseRequest/BaseResponse are Pydantic models. All model-specific requests/responses inherit from them. Validation at the boundary, not inside backends.
- **`@register_backend` decorator** — backends self-register by name. `ModelServer` looks up by name from `_BACKEND_REGISTRY`. Avoids circular imports (registry is a separate module).
- **Lazy imports inside `load()`** — optional dependencies (chronos, tabpfn, torch, etc.) are imported inside `load()`, not at module level. This keeps the core importable without any heavy deps installed.
- **`history` vs `feature_ref`** — `TimeSeriesRequest` accepts either raw float history or a Feast feature reference (mutually exclusive, validated by `@model_validator`).
- **Bolt vs Chronos2 inference** — `Chronos2Backend` handles both `ChronosBoltPipeline` (returns fixed 9 quantiles) and `Chronos2Pipeline` (returns samples). The distinction is detected at `load()` time via `isinstance` check.
- **TabPFN per-request fit** — TabPFN is an in-context learner. `batch_predict` runs each request independently (different context tables per request). Future: batch query rows against same context table.

## Adding a new backend

1. Add typed request/response to `src/sheaf/api/<model_type>.py`
2. Implement `ModelBackend` in `src/sheaf/backends/<name>.py`, decorated with `@register_backend("<name>")`
3. Add optional deps to `pyproject.toml` under `[project.optional-dependencies]`
4. Add tests in `tests/`

Reference implementation: `src/sheaf/backends/chronos.py` + `src/sheaf/api/time_series.py`

## Dev setup

```bash
git clone https://github.com/korbonits/sheaf.git
cd sheaf
uv sync --extra dev
uv run pytest tests/
```

Lint / format / type check:
```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run ty check src/          # ty (Astral's type checker, replaces mypy in CI)
uv run mypy src/sheaf          # mypy also configured
```

CI runs lint + tests on Python 3.10, 3.11, 3.12 via GitHub Actions.

## TabPFN requirement

`TABPFN_TOKEN` env var must be set before calling `TabPFNBackend.load()`. Obtain at https://ux.priorlabs.ai. First-run browser auth flow breaks in headless/container environments — token-only flow works.

## What's intentionally deferred

- Sphinx / mkdocs: deferred until v0.2 when the API surface stabilizes
- Async handlers: blocked on v0.2 Ray Serve integration
- Feast resolver: `feature_ref` field exists in `TimeSeriesRequest` but the resolver is not implemented
