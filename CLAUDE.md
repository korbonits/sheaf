# Sheaf — Claude Code Context

## What this project is

`sheaf-serve` is a unified serving layer for non-text foundation models (time series, tabular, molecular, geospatial, etc.). Think vLLM but for every model type that isn't a text LLM.

Each model type gets a typed request/response contract (Pydantic). Batching, caching, and scheduling are optimized per model type. Ray Serve is the execution substrate. Feast is a first-class input primitive.

PyPI: `pip install sheaf-serve`

## Current state: v0.2 (in progress)

**What works:**
- Time series: Chronos2 and TimesFM backends, full quantile/sample/mean output modes
- Tabular: TabPFN v2 backend, classification + regression
- Ray Serve integration end-to-end: `ModelServer.run()` deploys each `ModelSpec` as a Ray Serve deployment
- HTTP API: `GET /health`, `GET /ready`, `POST /predict` per deployment; 422 on bad input via Pydantic discriminated union
- Async inference: `ModelBackend.async_predict` / `async_batch_predict` run sync backends in a thread executor
- Batching: `@serve.batch` with `max_batch_size` and `timeout_ms` wired per deployment from `ModelSpec.batch_policy`
- Custom backends: `SHEAF_EXTRA_BACKENDS=mypackage.backends` imports extra backend modules in Ray workers at startup
- `backend_cls` field on `ModelSpec`: pass a class directly (cloudpickled) instead of a registry name

**Still needed before v0.2 is done:**
- Basic error handling at the service boundary (backend exceptions → structured 500, not actor crash)
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
  server.py            # ModelServer + _SheafDeployment — Ray Serve orchestrator
  registry.py          # @register_backend decorator + _BACKEND_REGISTRY dict
  api/
    base.py            # BaseRequest, BaseResponse, ModelType enum
    time_series.py     # TimeSeriesRequest/Response, Frequency, OutputMode
    tabular.py         # TabularRequest/Response
  backends/
    base.py            # ModelBackend ABC: load(), predict(), async_predict(), batch_predict()
    chronos.py         # Chronos2Backend — Chronos-Bolt + Chronos-T5 families
    tabpfn.py          # TabPFNBackend — TabPFN v2 classification + regression
    timesfm.py         # TimesFMBackend (exists, structure mirrors chronos.py)
  scheduling/
    batch.py           # BatchPolicy — wired into @serve.batch per deployment
  cache/               # stub
  integrations/        # stub
examples/
  quickstart.py        # Chronos time series example
  quickstart_tabular.py
  time_series_comparison.py  # Chronos vs TimesFM
tests/
  stubs.py             # Pytest-free stub backends for Ray worker cloudpickle
  test_api.py
  test_tabular_api.py
  test_server.py       # ModelBackend async dispatch, AnyRequest union, registry
  test_smoke_ray.py    # End-to-end Ray Serve tests (SHEAF_SMOKE_TEST=1 to run)
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
- **Registry populated inside `__init__`, not at module level** — Ray Serve may cloudpickle `_SheafDeployment` by value (inline), so module-level code in `server.py` may not run in workers. Standard backends and `SHEAF_EXTRA_BACKENDS` are imported inside `_SheafDeployment.__init__`, which always runs after `runtime_env.env_vars` are applied. The registry is also re-imported fresh there to avoid stale cloudpickle snapshots.
- **`@serve.batch` parameters set at runtime** — `max_batch_size` and `batch_wait_timeout_s` are fixed at class definition time by the decorator, but `@serve.batch` exposes `set_max_batch_size` / `set_batch_wait_timeout_s` setters. `_SheafDeployment.__init__` calls these to apply `ModelSpec.batch_policy` per deployment.
- **`model_type` fields use `Literal`** — request/response `model_type` fields are `Literal[ModelType.X]` not `ModelType`, which is required for Pydantic v2 discriminated unions.
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
uv run pre-commit install   # required once per clone — hooks mirror CI checks
uv run pytest tests/
```

Lint / format / type check (mirrors CI exactly):
```bash
uv run ruff check src/ tests/         # lint
uv run ruff format --check src/ tests/ # format check (use without --check to fix)
uv run ty check src/                  # type check
```

CI runs lint + tests on Python 3.10, 3.11, 3.12 via GitHub Actions.

## TabPFN requirement

`TABPFN_TOKEN` env var must be set before calling `TabPFNBackend.load()`. Obtain at https://ux.priorlabs.ai. First-run browser auth flow breaks in headless/container environments — token-only flow works.

## What's intentionally deferred

- Sphinx / mkdocs: deferred until the API surface stabilizes further
- Feast resolver: `feature_ref` field exists in `TimeSeriesRequest` but the resolver is not implemented
- `bucket_by` batching: `BatchPolicy.bucket_by` field exists but grouping requests by horizon (or other field) before batching is not yet implemented
- `BatchPolicy` via `ModelServer.run()`: batch parameters are wired in `__init__` via setters; there is no separate `.options()` API for this in Ray Serve
