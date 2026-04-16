# Sheaf — Claude Code Context

## What this project is

`sheaf-serve` is a unified serving layer for non-text foundation models (time series, tabular, molecular, geospatial, etc.). Think vLLM but for every model type that isn't a text LLM.

Each model type gets a typed request/response contract (Pydantic). Batching, caching, and scheduling are optimized per model type. Ray Serve is the execution substrate. Feast is a first-class input primitive.

PyPI: `pip install sheaf-serve`

## Current state: v0.2 (complete) / v0.3 (in progress)

**What works (v0.2):**
- Time series: Chronos2, TimesFM, and Moirai backends, full quantile/sample/mean output modes; multivariate support
- Tabular: TabPFN v2 backend, classification + regression
- Ray Serve integration end-to-end: `ModelServer.run()` deploys each `ModelSpec` as a Ray Serve deployment
- HTTP API: `GET /health`, `GET /ready`, `POST /predict` per deployment; 422 on bad input via Pydantic discriminated union
- Async inference: `ModelBackend.async_predict` / `async_batch_predict` run sync backends in a thread executor
- Batching: `@serve.batch` with `max_batch_size` and `timeout_ms` wired per deployment from `ModelSpec.batch_policy`
- Service-boundary error handling: backend exceptions → structured HTTP 500, actor does not crash
- Model hot-swap without restart: `ModelServer.update(spec)` does a rolling Ray Serve redeploy
- Custom backends: `SHEAF_EXTRA_BACKENDS=mypackage.backends` imports extra backend modules in Ray workers at startup
- `backend_cls` field on `ModelSpec`: pass a class directly (cloudpickled) instead of a registry name
- Container-friendly TabPFN auth: `load()` uses tabpfn's full token resolution order (env var → `~/.cache/tabpfn/auth_token` → `~/.tabpfn/token`); sets `TABPFN_NO_BROWSER=1` automatically; `TabPFNLicenseError` at fit-time is re-raised as `OSError`

**What works (v0.3 in progress):**
- Audio: Whisper backend (`openai-whisper`) and faster-whisper backend (`faster-whisper` / CTranslate2) — transcription, translation, word timestamps, VAD filter, language probability; WAV decoded inline (no ffmpeg needed for WAV inputs); install with `pip install 'sheaf-serve[audio]'`
- TTS: Bark backend (`suno/bark-small`, `suno/bark`) via HuggingFace `transformers.BarkModel` — text-to-speech with optional voice presets; outputs base64-encoded 16-bit PCM WAV at 24kHz; install with `pip install 'sheaf-serve[tts]'`
- Vision embeddings: OpenCLIP backend (`open-clip-torch`) — image and text embeddings via CLIP/SigLIP/EVA-CLIP; `EmbeddingRequest` accepts `texts` or `images_b64` (mutually exclusive); L2-normalized by default; install with `pip install 'sheaf-serve[vision]'`

**v0.3 remaining targets:**
- ESM-3 molecular backend
- GraphCast geospatial backend
- Feast feature resolver end-to-end
- TabPFN integration test (gated on `TABPFN_TOKEN`): real `load()` + `fit()` against the live library

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
    moirai.py          # MoiraiBackend — Salesforce Moirai (uni2ts)
    tabpfn.py          # TabPFNBackend — TabPFN v2 classification + regression
    timesfm.py         # TimesFMBackend
    whisper.py         # WhisperBackend — openai-whisper (PyTorch)
    faster_whisper.py  # FasterWhisperBackend — faster-whisper (CTranslate2, no torch at runtime)
    bark.py            # BarkBackend — Bark TTS via HuggingFace transformers
    open_clip.py       # OpenCLIPBackend — image/text embeddings via open-clip-torch
    _audio_utils.py    # Shared WAV encoding/decoding utility (no ffmpeg for WAV inputs)
  scheduling/
    batch.py           # BatchPolicy — wired into @serve.batch per deployment
  cache/               # stub
  integrations/        # stub
examples/
  quickstart.py        # Chronos time series example
  quickstart_tabular.py
  time_series_comparison.py  # Chronos vs TimesFM
  quickstart_audio.py        # Whisper + faster-whisper transcription, word timestamps, translation
  sample.wav                 # 4.8s 16kHz mono WAV for audio examples / smoke tests
tests/
  stubs.py             # Pytest-free stub backends for Ray worker cloudpickle
  test_api.py
  test_tabular_api.py
  test_server.py       # ModelBackend async dispatch, AnyRequest union, registry
  test_whisper_backend.py         # WhisperBackend mocked tests (8 tests)
  test_faster_whisper_backend.py  # FasterWhisperBackend mocked tests (9 tests)
  test_bark_backend.py            # BarkBackend mocked tests (9 tests)
  test_open_clip_backend.py       # OpenCLIPBackend mocked tests (12 tests)
  test_smoke_ray.py    # End-to-end Ray Serve tests (SHEAF_SMOKE_TEST=1 to run)
  test_smoke_whisper.py           # Whisper + faster-whisper e2e (SHEAF_SMOKE_TEST=1 to run)
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
- **faster-whisper lazy generator** — `WhisperModel.transcribe()` returns `(segments_generator, info)`. The generator must be fully consumed before `info` fields (language, duration) are reliable. `FasterWhisperBackend._run()` consumes it immediately in a list comprehension. Do not partially iterate.
- **WAV without ffmpeg** — `_audio_utils.decode_audio()` parses RIFF/WAV directly to float32 numpy at 16kHz for 16/32-bit PCM. Non-WAV formats fall back to a named temp file (calling backend passes the path; the model invokes ffmpeg internally).
- **WAV encoding** — `_audio_utils.encode_wav()` encodes a float32 numpy array to 16-bit PCM WAV bytes (pure numpy/struct, no scipy). Used by `BarkBackend` to produce the `audio_b64` response field.
- **TTS vs ASR model_type** — `TTSRequest`/`TTSResponse` use `ModelType.TTS = "tts"`, distinct from `ModelType.AUDIO = "audio"` used by Whisper/faster-whisper. Both are in `AnyRequest` discriminated union.
- **OpenCLIP mutually exclusive inputs** — `EmbeddingRequest` accepts either `texts: list[str]` or `images_b64: list[str]`, never both. Validated by `@model_validator`. A single request batches multiple items; `batch_predict` runs requests sequentially.
- **PIL stored at load() time** — `OpenCLIPBackend._Image` is set to `PIL.Image` during `load()` so tests can inject a mock without PIL installed in the test environment.

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
