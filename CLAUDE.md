# Sheaf — Claude Code Context

## What this project is

`sheaf-serve` is a unified serving layer for non-text foundation models (time series, tabular, molecular, geospatial, etc.). Think vLLM but for every model type that isn't a text LLM.

Each model type gets a typed request/response contract (Pydantic). Batching, caching, and scheduling are optimized per model type. Ray Serve is the execution substrate. Feast is a first-class input primitive.

PyPI: `pip install sheaf-serve`

## Current state: v0.7.0 (SheafWorker async job queue) shipped; v0.7 Track 2 (LoRA adapter multiplexing) next

Per-version ship notes live in git history and release tags. This doc tracks what exists *now* and the non-obvious design choices behind it. For feature-level changelog, see `git log`.

### Core serving
- **Ray Serve path** — `ModelServer.run()` deploys each `ModelSpec` as a `_SheafDeployment`. Endpoints per deployment: `GET /health`, `GET /ready`, `POST /predict`, `POST /{name}/stream` (SSE), `GET /metrics`. Hot-swap via `ModelServer.update(spec)`.
- **Modal serverless path** — `ModalServer` in `modal_server.py` is a zero-infra alternative; shared ASGI builder `_build_asgi_app`. No `@serve.batch`, so `bucket_by` is silently ignored.
- **Offline batch path** — `BatchRunner` + `BatchSpec` in `sheaf.batch` runs any backend over a JSONL source → JSONL sink via Ray Data `map_batches`. Two execution modes: `compute="tasks"` (default, stateless tasks with a worker-local `_BACKEND_CACHE`) or `compute="actors"` (warm `load()` per actor, sized by `num_actors`, for expensive backends like FLUX/GraphCast/SDXL).
- **Async-job worker path** — `SheafWorker` + `WorkerSpec` in `sheaf.worker` consume jobs from a queue; v1 backend is Redis Streams + consumer groups. `JobQueue` / `ResultStore` ABCs let SQS / Kafka / Postgres slot in later. `JobQueueClient.enqueue(req)` validates against `AnyRequest` and submits; results land in a Redis hash (`RedisHashResultStore`); optional per-job webhook POST on completion. At-least-once delivery (XACK only after result persistence); dead-letter stream on `max_retries` exhaustion.
- **Batching** — `@serve.batch` with per-deployment `max_batch_size` / `timeout_ms`; optional `BatchPolicy.bucket_by` sub-batches by a scalar field before dispatch.
- **Caching** — opt-in per deployment via `ModelSpec.cache = CacheConfig(...)`; in-process LRU, SHA-256 key, optional TTL. `SHEAF_CACHE_DISABLED=1` disables process-wide.
- **Feast integration** — `FeatureRef` on `TimeSeriesRequest`; `FeastResolver` wraps `feast.FeatureStore` and resolves online features per-request before batching/cache.
- **Ops/DX** — structured JSON logging (`SHEAF_LOG_JSON=1`), Prometheus metrics, OTel tracing (`sheaf.predict` → `sheaf.feast.resolve` + `sheaf.backend.infer`), SSE streaming.
- **Backend plugin model** — `@register_backend("name")` self-registration; or pass a class directly via `ModelSpec.backend_cls` (cloudpickled); or `SHEAF_EXTRA_BACKENDS=module1,module2` for worker-side discovery.

### Supported model types

| Model type | Backend(s) | API module | Extras flag |
|---|---|---|---|
| Time series | Chronos2, TimesFM, Moirai | `api/time_series.py` | (core) |
| Tabular | TabPFN v2 | `api/tabular.py` | (core) |
| Audio (ASR) | Whisper, faster-whisper | `api/audio.py` | `[audio]` |
| Audio generation | MusicGen | `api/audio_generation.py` | `[audio-generation]` |
| TTS | Bark, Kokoro | `api/audio.py` (TTSRequest) | `[tts]`, `[kokoro]` |
| Vision embedding | OpenCLIP, DINOv2 | `api/embedding.py` | `[vision]` |
| Segmentation | SAM2 | `api/segmentation.py` | `[vision]` |
| Depth | Depth Anything v2 | `api/depth.py` | `[vision]` |
| Object detection | DETR/RT-DETR | `api/detection.py` | `[vision]` |
| Pose | ViTPose | `api/pose.py` | `[pose]` |
| Optical flow | RAFT | `api/optical_flow.py` | `[optical-flow]` |
| Video | VideoMAE / TimeSformer | `api/video.py` | `[video]` |
| Image diffusion | FLUX (schnell/dev) | `api/diffusion.py` | `[diffusion]` |
| Multimodal generation | SDXL img2img/inpaint | `api/multimodal_generation.py` | `[multimodal-generation]` |
| Cross-modal embedding | ImageBind | `api/multimodal_embedding.py` | `[multimodal]` (+ git install) |
| Molecular (protein) | ESM-3 (Py 3.12+) | `api/molecular.py` | `[molecular]` |
| Genomics | Nucleotide Transformer | `api/genomic.py` | `[genomics]` |
| Small molecule | MolFormer | `api/small_molecule.py` | `[small-molecule]` |
| Materials | MACE-MP-0 | `api/materials.py` | `[materials]` |
| Weather | GraphCast | `api/weather.py` | `[weather]` |
| Earth observation | Prithvi-EO | `api/satellite.py` | `[earth-observation]` |
| LiDAR / 3D | PointNet (pure PyTorch) | `api/point_cloud.py` | `[lidar]` |

Other extras: `[feast]`, `[modal]`, `[metrics]`, `[tracing]`, `[batch]`.

## Repo layout

```
src/sheaf/
  __init__.py          # public exports: ModelServer, ModelSpec
  spec.py              # ModelSpec, ResourceConfig
  server.py            # ModelServer + _SheafDeployment (Ray Serve orchestrator)
  modal_server.py      # ModalServer (Modal serverless alternative)
  registry.py          # @register_backend + _BACKEND_REGISTRY
  logging.py           # JsonFormatter + configure_logging (SHEAF_LOG_JSON=1)
  metrics.py           # Prometheus: record_predict/batch/load, register_metrics_endpoint
  tracing.py           # OTel: configure_tracing, trace_predict, trace_span, record_exception
  api/                 # one module per model type; see table above. union.py = AnyRequest
  backends/            # one module per backend (name matches the table above)
    base.py            # ModelBackend ABC: load/predict/async_predict/batch_predict/stream_predict
    _register.py       # register_builtin_backends() + register_extra_backends() for workers
    _audio_utils.py    # WAV encode/decode (no ffmpeg for WAV inputs)
  scheduling/batch.py  # BatchPolicy + bucket_requests()
  cache/               # CacheConfig + ResponseCache (LRU + optional TTL)
  integrations/feast.py  # FeastResolver (wraps feast.FeatureStore)
  batch/               # BatchRunner, BatchSpec, JsonlSource, JsonlSink
  worker/              # SheafWorker, WorkerSpec, JobQueue/ResultStore ABCs, RedisStreamsQueue, RedisHashResultStore, JobQueueClient
examples/              # quickstart_*.py per model type + sample.wav
tests/                 # test_<backend>_backend.py (mocked) + test_smoke_*.py (gated on SHEAF_SMOKE_TEST=1)
                       # test_tabpfn_integration.py gated on TABPFN_TOKEN
                       # stubs.py: pytest-free stub backends for Ray worker cloudpickle
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
- **Container-friendly TabPFN auth** — `load()` uses tabpfn's full token resolution order (env var → `~/.cache/tabpfn/auth_token` → `~/.tabpfn/token`); sets `TABPFN_NO_BROWSER=1` automatically; `TabPFNLicenseError` at fit-time is re-raised as `OSError`.
- **faster-whisper lazy generator** — `WhisperModel.transcribe()` returns `(segments_generator, info)`. The generator must be fully consumed before `info` fields (language, duration) are reliable. `FasterWhisperBackend._run()` consumes it immediately in a list comprehension. Do not partially iterate.
- **WAV without ffmpeg** — `_audio_utils.decode_audio()` parses RIFF/WAV directly to float32 numpy at 16kHz for 16/32-bit PCM. Non-WAV formats fall back to a named temp file (calling backend passes the path; the model invokes ffmpeg internally).
- **WAV encoding** — `_audio_utils.encode_wav()` encodes a float32 numpy array to 16-bit PCM WAV bytes (pure numpy/struct, no scipy). Used by `BarkBackend` and `MusicGenBackend` to produce the `audio_b64` response field.
- **TTS vs ASR vs audio generation model_type** — `TTSRequest`/`TTSResponse` use `ModelType.TTS = "tts"`; `AudioGenerationRequest`/`AudioGenerationResponse` use `ModelType.AUDIO_GENERATION = "audio_generation"`; Whisper/faster-whisper use `ModelType.AUDIO = "audio"`. All three are in `AnyRequest` discriminated union.
- **MusicGen frame rate** — `max_new_tokens = int(duration_s * model.config.audio_encoder.frame_rate)` (frame_rate=50 tokens/sec for MusicGen). Audio output shape is `(1, n_channels, T)`; `[0]` gives `(n_channels, T)`; mono: `[0][0]`; stereo: `mean(axis=0)`.
- **OpenCLIP mutually exclusive inputs** — `EmbeddingRequest` accepts either `texts: list[str]` or `images_b64: list[str]`, never both. Validated by `@model_validator`. A single request batches multiple items; `batch_predict` runs requests sequentially.
- **PIL stored at load() time** — `OpenCLIPBackend._Image` and `DINOv2Backend._Image` are set to `PIL.Image` during `load()` so tests can inject a mock without PIL installed in the test environment.
- **DINOv2 pooling strategies** — `DINOv2Backend` supports `pooling="cls"` (CLS token at `last_hidden_state[:, 0, :]`, default) and `pooling="mean"` (mean of patch tokens at `[:, 1:, :]`). Image-only; raises `ValueError` on `texts` input.
- **ESM-3 pooling strategies** — `ESM3Backend` supports `pooling="mean"` (mean of residue hidden states at positions `1:-1`, excluding BOS/EOS, default) and `pooling="cls"` (BOS token at position 0). Requires Python 3.12+ and HuggingFace Hub login with EvolutionaryScale license accepted.
- **Nucleotide Transformer 6-mer tokenization** — sequences are processed one at a time; mean pooling excludes CLS (pos 0) and EOS (pos -1) via `hidden[:, 1:-1, :]`.
- **MolFormer attention-masked mean pooling** — tokenizes the entire batch at once with `padding=True`; masked mean: `(hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1)` where `mask_exp = attention_mask.unsqueeze(-1).float()`.
- **MACE ASE Atoms interface** — `self._Atoms` is stored at `load()` time (like `self._Image` in vision backends) for test injectability. Tests inject `FakeAtoms` class that captures constructor kwargs.
- **SAM2 mask encoding** — `SegmentationResponse.masks_b64` are base64-encoded flat uint8 byte arrays. Decode with `np.frombuffer(base64.b64decode(m), dtype=np.uint8).reshape(height, width).astype(bool)`.
- **Depth Anything normalization** — `DepthRequest.normalize=True` (default) maps raw depth to `[0, 1]` via `(d - min) / range`. `DepthResponse.depth_b64` is a base64-encoded flat float32 byte array; reshape with `np.frombuffer(..., dtype=np.float32).reshape(height, width)`.
- **DETR target_sizes ordering** — PIL `.size` returns `(W, H)` but `post_process_object_detection` requires `target_sizes=[(H, W)]`. The backend unpacks as `img_width, img_height = img.size` and reverses for the call.
- **`_real_import` pattern in tests** — when patching `builtins.__import__` to block a specific module, capture `_real_import = builtins.__import__` first and delegate all non-blocked imports to it. Avoids infinite recursion when the backend's `load()` does `from PIL import Image` (which re-enters `__import__`). Inject any needed mocks via `sys.modules` instead of relying on `__import__` for them.
- **Prithvi `trust_remote_code=True`** — `PrithviBackend` passes `trust_remote_code=True` to both `AutoModel.from_pretrained` and `AutoImageProcessor.from_pretrained`. `self._processor` is stored at `load()` time (like `self._Image` in vision backends) for test injectability.
- **Prithvi normalization** — `_normalize()` broadcasts processor `image_mean`/`image_std` over `(n_time, n_bands, H, W)` axis=1. Skips silently if processor lacks those attrs (`AttributeError`/`TypeError`) or if band count mismatches.
- **GraphCast checkpoint** — `GraphCastBackend.load()` opens `checkpoint_path` with `pickle.load` (the `.npz` checkpoint is actually a pickle). Haiku `transform_with_state` is JIT-compiled at load time; tests patch `builtins.open` with `mock_open()` to avoid FileNotFoundError.
- **SatelliteRequest pixels layout** — `pixels_b64` encodes a `(n_time, n_bands, H, W)` float32 array. `PrithviBackend` decodes, optionally normalizes, then unsqueezes to `(1, n_time, n_bands, H, W)` before the model call.
- **ImageBind not on PyPI** — `imagebind-packaged` pins `torch==1.13.0` (2022) and is incompatible with any modern ML stack. Install from source: `pip install git+https://github.com/facebookresearch/ImageBind.git`. The `[multimodal]` extra provides `torch>=2.0.0` and `torchvision`/`torchaudio`. A clear `ImportError` is raised at `load()` time with installation instructions.
- **ImageBind temp files** — imagebind data loaders accept file paths, not in-memory buffers. `ImageBindBackend._build_inputs()` writes base64-decoded bytes to `tempfile.mkstemp()` files and passes the paths to loaders. `_remove_files()` is called in a `finally` block after inference, so cleanup happens even on exceptions.
- **ImageBind modality exclusivity** — `MultimodalEmbeddingRequest` accepts exactly one of: `texts`, `images_b64`, `audios_b64`, `depth_images_b64`, `thermal_images_b64`. Validated by `@model_validator`. A single request batches all items within a modality.
- **ImageBind MagicMock side_effect pattern** — in tests, `model.side_effect = _forward` (not `model.__call__ = ...`) is the correct way to override MagicMock call behavior. Python looks up special methods on the type, not the instance — setting `__call__` on a MagicMock instance has no effect.
- **FLUX pipeline call chain** — `FluxPipeline.from_pretrained(model_id, torch_dtype=...).to(device)` returns the pipeline instance. Calling `pipeline(prompt=..., ...)` returns an object with `.images[0]` being a PIL image. Mock chain in tests: `pipeline_cls.from_pretrained.return_value.to.return_value = instance` and `instance.return_value.images = [fake_img]`.
- **FLUX dtype resolution** — `torch_dtype` is stored as a string (`"bfloat16"`) at `__init__` time; resolved to the actual `torch.dtype` inside `load()` via a dict lookup. This avoids importing torch at construction time.
- **FLUX CPU offload** — when `enable_model_cpu_offload=True`, `pipeline.enable_model_cpu_offload()` is called on the pre-`.to()` pipeline object (not the instance). `.to(device)` is skipped in this path. Tests assert `instance.to.assert_not_called()`.
- **VideoMAE frame input** — `VideoRequest.frames_b64` is a list of base64-encoded PNG/JPEG frames (min_length=1); validated at request boundary. `VideoMAEBackend._run()` decodes each frame via `PIL.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")` and passes the list to `AutoImageProcessor`.
- **VideoMAE pooling strategies** — CLS token at `hidden[:, 0, :]` (default); mean of patch tokens at `hidden[:, 1:, :]` when `pooling="mean"`. L2-normalized if `request.normalize=True`.
- **VideoMAE classification num_classes** — `logits.shape[-1]` must be an int for `min(5, num_classes)`. When mocking, set `fake_output.logits.shape = (1, 5)` as a real Python tuple (not a MagicMock attribute), otherwise `min()` comparison raises TypeError.
- **VideoMAE `_Image` at load time** — `self._Image = _Image` is stored after `from PIL import Image as _Image` inside `load()`, following the same pattern as `DINOv2Backend`. Tests inject a mock without requiring PIL installed.
- **Modal local-source pattern for v0.4** — `pip_install_from_pyproject` installs deps listed in `pyproject.toml` extras but does NOT install the sheaf package itself. Pre-PyPI release: use `.add_local_dir("src", remote_path="/root/src", copy=True).env({"PYTHONPATH": "/root/src"})`. `copy=True` is required so the `.env()` build step can follow. When v0.4 ships to PyPI, revert to `pip install 'sheaf-serve[video]>=0.4.0'`.
- **Prometheus metrics — `register_metrics_endpoint` deferred in Ray Serve** — `@serve.ingress(_app)` cloudpickles `_app` at class-definition time. Registering the `/metrics` route at module level causes a `RecursionError`: cloudpickle follows the nested function's `__globals__` into `sheaf.metrics`, which holds live `CollectorRegistry` and metric objects containing `threading.Lock` (unpicklable). Fix: call `register_metrics_endpoint(_app, spec.name)` inside `_SheafDeployment.__init__`. The worker deserialises `_app` before `__init__` runs, so subsequent route additions are safe and never re-pickled.
- **Prometheus metrics — lazy `/metrics` handler** — the `metrics()` handler in `register_metrics_endpoint` imports `prometheus_client` and calls `_get_registry()` at request time (not at route-registration time). This keeps the function's closure free of non-picklable objects, which is required by the Ray Serve path above.
- **Structured logging** — `sheaf.logging.JsonFormatter` emits single-line JSON with `ts`, `level`, `logger`, `message`, and any `extra={}` fields. `configure_logging()` installs it idempotently on the root logger; gated by `SHEAF_LOG_JSON=1`. Both `server.py` and `modal_server.py` log `request_id`, `deployment`, `model_type`, `model_name`, `latency_ms`, `status` ("ok"/"error"), and `error` (on failures) via `logger.info/exception("predict ok/error", extra={...})`.
- **Prometheus metrics names** — `sheaf_requests_total` (Counter, labels: deployment/model_type/status), `sheaf_request_duration_seconds` (Histogram), `sheaf_batch_size_total` (Histogram, labels: deployment), `sheaf_backend_load_seconds` (Gauge). `SHEAF_METRICS_DISABLED=1` disables all recording and endpoint registration even when `prometheus_client` is installed.
- **OTel tracing singleton** — `trace.set_tracer_provider()` is a one-time operation per process; subsequent calls are silently ignored by OTel. `configure_tracing()` is idempotent by checking `isinstance(current, SdkTracerProvider)` before setting. Tests that test span creation use `monkeypatch.setattr(t, "get_tracer", lambda: local_provider.get_tracer("sheaf"))` with a `TracerProvider` + `InMemorySpanExporter` to bypass global state; tests that verify `configure_tracing()` mock `trace.set_tracer_provider` and `trace.get_tracer_provider` instead.
- **OTel tracing spans** — `sheaf.predict` outer span wraps the full request; sub-spans `sheaf.feast.resolve` and `sheaf.backend.infer` are emitted via `trace_span()`. `trace_predict()` does not auto-record exceptions (call `record_exception(span, exc)` explicitly in the `except` block before raising `HTTPException`) so the original exception is captured, not the wrapping HTTP exception. `record_exception()` guards the `StatusCode` import so call sites don't need OTel installed.
- **OTel `configure_tracing()` env vars** — `OTEL_EXPORTER_OTLP_ENDPOINT` triggers OTLP/HTTP exporter (appends `/v1/traces`); `SHEAF_OTEL_CONSOLE=1` triggers `ConsoleSpanExporter` (dev/CI). If neither is set, `configure_tracing()` returns immediately without initialising the SDK (zero overhead). `OTEL_SERVICE_NAME` overrides the service name resource attribute.
- **`bucket_by` batching — sub-batch within a Ray Serve batch window** — `BatchPolicy.bucket_by` names a scalar field (e.g. `"horizon"` for time series, `"n_frames"` for video). `bucket_requests(requests, bucket_by)` in `scheduling/batch.py` groups the list into `(original_indices, sub_requests)` pairs, one per unique field value, preserving arrival order within each bucket. `_batch_predict` dispatches one `async_batch_predict` call per group and reassembles results in original order via a `slot: dict[int, ...]` index map. `modal_server.py` handles requests one at a time (no `@serve.batch`), so `bucket_by` has no effect there and is silently ignored. Requests lacking the field are grouped under the `None` bucket.
- **Request cache — opt-in per deployment** — `ModelSpec.cache = CacheConfig(enabled=True)` attaches an in-process LRU `ResponseCache` to that deployment. Default is `enabled=False` (no overhead). `SHEAF_CACHE_DISABLED=1` skips all caches process-wide regardless of spec config (useful in integration test runs where you want to exercise the backend every time).
- **Cache key design** — `ResponseCache.make_key(deployment, request)` serialises the request via `model_dump(mode="json", exclude={"request_id", ...})`, prepends the deployment name (so two specs with identical input types share no entries), JSON-canonicalises with `sort_keys=True`, then SHA-256 hashes. `request_id` is always excluded (it is unique per call). Additional fields can be excluded via `CacheConfig.exclude_fields` (e.g. `["seed"]` for diffusion models where different seeds should be distinct cache entries but you still want same-seed repeats to hit).
- **Cache placement — after Feast resolution** — the cache lookup and store happen after Feast feature resolution, inside the `trace_predict` span, before `_batch_predict`. This means the cache key reflects the actual history values (not a feature reference), so two requests for the same entity key at different times (with different resolved feature values) correctly produce distinct entries. The same placement applies in `modal_server.py`'s predict handler.
- **Cache store — dict, not response object** — `ResponseCache` stores `dict[str, Any]` (the `model_dump(mode="json")` output), not the Pydantic response object. This matches what the predict handler returns to FastAPI and avoids a second serialisation round-trip on cache hits.
- **`ResponseCache` is never cloudpickled** — `self._cache` is assigned in `_SheafDeployment.__init__`, which runs after `@serve.ingress` deserialises `_app` in the worker. Unlike `register_metrics_endpoint`, there is no cloudpickle hazard here.
- **`stream_predict` async generator** — `ModelBackend.stream_predict(request)` is an async generator that yields event dicts. Default implementation yields one `{"type": "result", "done": True, ...response}` event after `async_predict()` completes. Backends override this for chunked output.
- **FLUX streaming via `threading.Queue`** — `FluxBackend.stream_predict` runs `_run()` in a thread-pool executor via `loop.run_in_executor()`. A `queue.Queue` (thread-safe) bridges the synchronous `callback_on_step_end` (called in the executor thread) to the async generator (running in the event loop thread). Progress events are drained via `q.get_nowait()` / `asyncio.sleep(0.02)` polling while `future.done()` is False, then drained once more after the future completes to avoid race conditions.
- **`POST /{name}/stream` bypasses batching and cache** — streaming responses are per-request by nature; they cannot be batched (the SSE body is produced incrementally) or meaningfully cached (the stream is ephemeral). The endpoint Feast-resolves identically to `/predict`, then calls `backend.stream_predict()` directly. Errors mid-stream yield a `{"type": "error", "error": "..."}` event instead of crashing the SSE response.
- **SSE wire format** — each event is a `data: {json}\n\n` line (standard HTML Server-Sent Events spec). Clients parse by stripping the `data: ` prefix and JSON-decoding the rest.
- **Kokoro reuses TTSRequest/TTSResponse** — `KokoroBackend` shares the same `TTSRequest`/`TTSResponse` contract as `BarkBackend`. Voice preset and speed are per-request fields on `TTSRequest`. `self._pipeline` is assigned in `load()` from `kokoro.KPipeline(lang_code)`, where `lang_code` defaults to `"a"` (American English).
- **ViTPose top-down + `_keypoint_names()`** — ViTPose is a top-down estimator: it runs on person crops, not the full image. `PoseRequest.bboxes` is `list[list[float]]` (one `[x1,y1,x2,y2]` per person); if empty, the full image is used as a single crop. `AutoProcessor` expects boxes as `[[[x1,y1,x2,y2], ...]]` (one list-of-boxes per image). `_keypoint_names()` reads `model.config.id2label` and returns a list indexed by integer key; falls back to `[]` on `AttributeError`/`KeyError`.
- **RAFT padding-crop and flow sign** — RAFT requires input dimensions to be multiples of 8. `RAFTBackend._run()` pads both frames to the next multiple-of-8 boundary with `F.pad`, calls RAFT, then crops the output flow field back to `(original_H, original_W)`. Flow sign convention: dx > 0 means pixels moved right (frame2 content is to the right of frame1). A checkerboard shifted 16px rightward in frame2 produces `dx ≈ -16` (content in frame1 moved leftward to reach frame2). `self._transforms` is stored at `load()` time for testability (same pattern as `self._Image`).
- **SDXL mode at `__init__` time** — `SDXLBackend.__init__` takes `mode: str` ("img2img" or "inpaint"); the corresponding pipeline class (`StableDiffusionXLImg2ImgPipeline` or `StableDiffusionXLInpaintPipeline`) is selected at `load()`. `torch_dtype` is a string at `__init__` and resolved to `torch.dtype` inside `load()` (same pattern as `FluxBackend`). `negative_prompt` is omitted from pipeline kwargs entirely when the request field is empty/None — passing `negative_prompt=""` alters conditioning vs. omitting it.
- **PointNet `_build_pointnet` module-level function** — the PointNet architecture is defined inside a module-level `_build_pointnet(num_classes)` function (not inside a class), allowing the inner `_PointNetModel` class to close over `num_classes` cleanly. All three torch imports inside `_build_pointnet()` (`torch`, `torch.nn`, `torch.nn.functional`) require `# ty: ignore[unresolved-import]` — CI runs without torch installed, and ty flags unresolved imports inside module-level functions as hard errors (unlike inside methods). `self._F` (torch.nn.functional) is stored at `load()` for testability; `self._model` is the built + loaded network.
- **BatchRunner — stateless tasks + worker-local module cache** — default `compute="tasks"` runs `ds.map_batches(_infer, batch_format="pandas", ...)` in stateless task mode. Ray Data reuses worker processes across tasks within a single job, so a module-level `_BACKEND_CACHE: dict[str, ModelBackend]` keyed by `spec.name` means `load()` fires once per worker (not once per batch). Best for cheap loads and small jobs.
- **BatchRunner — actor-pool mode for expensive `load()`** — opt-in via `BatchSpec.compute="actors"` + `num_actors=N`. Switches to `ds.map_batches(_BackendActor, concurrency=N, ...)` where `_BackendActor.__init__` calls `_build_backend(spec)` once per actor and `__call__` runs `batch_predict` per batch. The loaded backend persists for the actor's lifetime, eliminating the per-batch worker-cache fallback in the task path. `concurrency=N` (modern Ray Data API, ≥2.6) is preferred over the older `compute=ActorPoolStrategy(size=N)`. Sizing constraint: `num_actors * num_gpus <= cluster GPUs`. The `_BackendActor` class is defined inside `run()` so it closes over `captured_spec`; Ray cloudpickles the class for actor instantiation. Cold-start cost (first batch per actor blocks on `load()`) is the price of the warm-load guarantee — pre-warm with a dummy batch if it matters. Ray Data terminates actors when the dataset is fully consumed (`take_all()`); driver crashes can leave actors lingering, no per-run cleanup beyond Ray's own GC.
- **BatchRunner — driver-side pre-validation** — rows are pre-validated against `TypeAdapter(AnyRequest)` and their `model_type` checked against `spec.model_type` on the driver *before* `ray.data.from_items()`. Users see schema errors up-front rather than halfway through a long distributed job. The inner `_infer` also re-validates each pandas row dict before the backend call (belt-and-suspenders, and cheap).
- **BatchRunner — worker registry refresh** — Ray Data workers don't inherit the driver's import state, and module-level imports may be stale cloudpickle snapshots. `_build_backend` calls `register_builtin_backends()` + `register_extra_backends()` on every worker and re-imports `_BACKEND_REGISTRY as _registry` fresh *after* those calls. Custom / test-only backends registered via `SHEAF_EXTRA_BACKENDS=module1,module2` reach workers this way.
- **BatchRunner — `uv run` + Ray Data workers** — Ray's `uv_runtime_env_hook.py` parses the `uv run` CLI and packages the working dir as the worker env, but it does *not* forward `--extra` flags transitively. Running `uv run python examples/quickstart_batch.py` without `--extra batch` yields `ModuleNotFoundError: pandas` inside workers. Fix: `uv run --extra batch python ...`, or activate the venv directly (`source .venv/bin/activate`). CI is safe because `uv sync --extra batch` populates the venv before `uv run pytest`.
- **BatchRunner — `AnyRequest` lives in `sheaf.api.union`** — originally defined in `server.py` alongside Ray Serve imports. Moved into its own module so `batch.runner` can import it without pulling `ray.serve` (the `[batch]` extra depends only on `ray[default]`, pandas, pyarrow — not on `ray[serve]`). `server.py` now just does `from sheaf.api.union import AnyRequest`.
- **BatchRunner — pandas `batch_format` required** — Sheaf requests are nested dicts (e.g. `history: list[float]`, `bboxes: list[list[float]]`) that don't fit Ray Data's default numpy batch format cleanly. `batch_format="pandas"` converts each batch to a `pd.DataFrame`; `pdf.to_dict(orient="records")` then gives us back the per-row dicts we can validate via `TypeAdapter`.
- **BatchRunner — ty invalid-argument-type on `map_batches`** — Ray's `UserDefinedFunction` type is too broad for ty to prove the `DataFrame → DataFrame` signature of `_infer`. The `_infer` argument to `ds.map_batches` carries `# ty: ignore[invalid-argument-type]`. Same for the `import pandas` / `import pyarrow` / `import ray.data` lines inside `run()` — they carry `# ty: ignore[unresolved-import]` because CI linters run without the `[batch]` extra installed.
- **BatchRunner — explicit row-index sort, not Ray Data lineage** — earlier versions assumed `map_batches` + `take_all()` preserved input order; that is **not** true under Ray Data's streaming executor (default in Ray 2.x), which CI caught on Python 3.12 with `[req-006, req-000..req-005]` (7 rows / batch_size=3). The runner now injects a `_sheaf_row_idx` sentinel column on `from_items`, propagates it through `_run_batch` (popped before request validation, re-attached on each response record), and sorts + drops it after `take_all()`. Works identically across `compute="tasks"` and `compute="actors"`. `_run_batch` lazy-imports pandas inside the function (vs at module level) so unrelated imports don't pull `[batch]` extra deps.
- **SheafWorker — at-least-once via XACK-after-persist** — the consume loop is `dequeue → predict → results.put → webhook → ack`. XACK is the *only* signal Redis Streams uses to remove a job from the consumer group's pending-entry-list, so a worker crash between `predict` and `ack` causes the job to be redelivered to another consumer. This guarantees at-least-once; backend code that produces side-effects (e.g. database writes) must be idempotent or the result hash must be the source of truth.
- **SheafWorker — local delivery counter, not PEL** — Redis Streams' XPENDING gives true delivery count via the pending-entry-list, but `JobQueue` is an ABC and not every backend (SQS, Kafka) exposes that cleanly. `SheafWorker` keeps a process-local `dict[job_id, int]` counter and increments on each `dequeue` of the same id. Trade-off: a worker process restart loses the counter, so a job that has already burned its retries on worker A starts fresh on worker B. Acceptable for v1 since `max_retries` is bounded and dead-letter writes a `status="failed"` result either way.
- **SheafWorker — dead-letter writes a "failed" JobResult** — when retries are exhausted, the worker XADDs to the dead-letter stream *and* writes a `JobResult(status="failed", error=...)` to the result store. Without the result-store write, `JobQueueClient.wait_for_result(job_id, timeout_s)` would block until timeout for every poison-pill job. The dead-letter stream is for ops/triage; the result store is for waiting clients.
- **SheafWorker — webhook is best-effort** — webhook POST happens *between* `results.put` and `queue.ack`, but failures are caught and logged, not raised. The result is already in the store by then; downstream can poll. If we let webhook failures bubble, a flaky callback URL would dead-letter otherwise-successful jobs.
- **SheafWorker — model_type mismatch is non-retriable in practice** — the worker validates `request.model_type == spec.model_type` inside `_process` (after dequeue, before predict). A mismatch raises `ValueError` and falls through the same retry path as backend errors — but the failure is deterministic (the request will mismatch on every redelivery), so retries are wasteful. Tests use `max_retries=1` to dead-letter mismatch jobs immediately. Future: add a `is_permanent_failure(exc) -> bool` hook to short-circuit retries on validation errors.
- **SheafWorker — `record_predict` with deployment label** — the worker reuses `sheaf.metrics.record_predict(spec.name, model_type, status, latency)` rather than introducing a separate `record_job` family. Same Prometheus series shape (`sheaf_requests_total{deployment, model_type, status}`); workers are distinguished from Ray Serve deployments only by the `deployment` label value. Avoids dashboard drift.
- **SheafWorker — Redis import is lazy + injectable** — `RedisStreamsQueue.__init__` accepts a `client=` parameter for test injection; in production it lazy-imports `redis` and calls `Redis.from_url(url, decode_responses=True)`. The `decode_responses=True` matters: without it, every `xadd` field comes back as `bytes` and the `Job` model's `request: dict` validation fails on stringified bytes. Same pattern in `RedisHashResultStore`.
- **SheafWorker — `XGROUP CREATE ... MKSTREAM`** — `_ensure_group` is called from `__init__`, creating both the stream (if missing) and the consumer group atomically. Catches `BUSYGROUP` to make it idempotent — multiple workers calling `__init__` against the same stream/group don't fight. Required because `XREADGROUP` errors out if the group doesn't exist; we don't want every worker boot to require a manual setup step.
- **SheafWorker — `signal.signal` only on main thread** — graceful shutdown handlers are installed in `_install_signal_handlers` but wrapped in `try/except ValueError`. Python's `signal.signal` raises `ValueError` outside the main thread; the `except` lets the worker run inside a test thread or a thread-based supervisor without crashing. In a real deployment the worker is the main process anyway, so the handlers wire up.
- **SheafWorker — `JobQueueClient.enqueue` validates up-front** — the client calls `TypeAdapter(AnyRequest).validate_python(req)` before `queue.enqueue`. Schema errors surface at submission time (where the application code that constructed the bad request can handle them), not in a worker log fifteen minutes later. Mirrors the same driver-side pre-validation pattern in `BatchRunner.run`.

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

CI runs lint + tests on Python 3.11, 3.12 via GitHub Actions.

## TabPFN requirement

`TABPFN_TOKEN` env var must be set before calling `TabPFNBackend.load()`. Obtain at https://ux.priorlabs.ai. First-run browser auth flow breaks in headless/container environments — token-only flow works.

## What's intentionally deferred

- Sphinx / mkdocs: deferred until the API surface stabilizes further
- `BatchPolicy` via `ModelServer.run()`: batch parameters are wired in `__init__` via setters; there is no separate `.options()` API for this in Ray Serve
