# Sheaf — Claude Code Context

## What this project is

`sheaf-serve` is a unified serving layer for non-text foundation models (time series, tabular, molecular, geospatial, etc.). Think vLLM but for every model type that isn't a text LLM.

Each model type gets a typed request/response contract (Pydantic). Batching, caching, and scheduling are optimized per model type. Ray Serve is the execution substrate. Feast is a first-class input primitive.

PyPI: `pip install sheaf-serve`

## Current state: v0.4 complete / v0.5 next

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

**What works (v0.3):**
- Audio: Whisper backend (`openai-whisper`) and faster-whisper backend (`faster-whisper` / CTranslate2) — transcription, translation, word timestamps, VAD filter, language probability; WAV decoded inline (no ffmpeg needed for WAV inputs); install with `pip install 'sheaf-serve[audio]'`
- Audio generation: MusicGen backend (`facebook/musicgen-*`) via HuggingFace `transformers.MusicgenForConditionalGeneration` — text-conditioned music/audio generation; `AudioGenerationRequest` → `AudioGenerationResponse`; mono/stereo support; `max_new_tokens = int(duration_s * frame_rate)`; outputs base64-encoded 16-bit PCM WAV at 32kHz; install with `pip install 'sheaf-serve[audio-generation]'`
- TTS: Bark backend (`suno/bark-small`, `suno/bark`) via HuggingFace `transformers.BarkModel` — text-to-speech with optional voice presets; outputs base64-encoded 16-bit PCM WAV at 24kHz; install with `pip install 'sheaf-serve[tts]'`
- Vision embeddings: OpenCLIP backend (`open-clip-torch`) — image and text embeddings via CLIP/SigLIP/EVA-CLIP; `EmbeddingRequest` accepts `texts` or `images_b64` (mutually exclusive); L2-normalized by default; install with `pip install 'sheaf-serve[vision]'`
- Vision embeddings: DINOv2 backend (`transformers`) — image-only CLS or mean-pooled embeddings; install with `pip install 'sheaf-serve[vision]'`
- Segmentation: SAM2 backend (`sam2`) — prompted image segmentation via point coords, labels, and/or bounding boxes; returns base64-encoded uint8 masks; install with `pip install 'sheaf-serve[vision]'`
- Depth estimation: Depth Anything v2 backend (`transformers`) — monocular depth estimation; returns base64-encoded float32 depth map + min/max; install with `pip install 'sheaf-serve[vision]'`
- Object detection: DETR/RT-DETR backend (`transformers`) — any `AutoModelForObjectDetection`-compatible model; returns boxes in `[x1,y1,x2,y2]` pixel coords, class labels, scores; install with `pip install 'sheaf-serve[vision]'`
- Molecular embeddings: ESM-3 backend (`esm>=3.0.0`) — protein sequence embeddings via EvolutionaryScale ESM-3; CLS or mean pooling; **Python 3.12+ required**; install with `pip install 'sheaf-serve[molecular]'`
- Genomics embeddings: Nucleotide Transformer backend (`transformers`) — DNA/RNA sequence embeddings via InstaDeepAI/EMBL-EBI Nucleotide Transformer v2; 6-mer tokenization; mean pooling excludes CLS/EOS; install with `pip install 'sheaf-serve[genomics]'`
- Small molecule embeddings: MolFormer backend (`transformers`, `trust_remote_code=True`) — SMILES embeddings via IBM MolFormer-XL; batched tokenization with attention-masked mean pooling; install with `pip install 'sheaf-serve[small-molecule]'`
- Materials science: MACE backend (`mace-torch`) — universal interatomic potential via MACE-MP-0; energy, forces, and optional stress via ASE `Atoms` interface; install with `pip install 'sheaf-serve[materials]'`
- Weather forecasting: GraphCast backend (`graphcast`, `dm-haiku`, `jax`, `xarray`) — autoregressive n-step rollout from ERA5 surface+atmospheric fields; checkpoint loaded from `.npz`; install with `pip install 'sheaf-serve[weather]'`
- Earth observation: Prithvi backend (`transformers`, `torch`) — IBM/NASA Prithvi-EO geospatial embeddings via `AutoModel` + `AutoImageProcessor` with `trust_remote_code=True`; `SatelliteRequest` accepts `(n_time, n_bands, H, W)` float32 pixels; per-band z-score normalization; CLS and mean pooling; install with `pip install 'sheaf-serve[earth-observation]'`
- Cross-modal embeddings: ImageBind backend — five modalities (text, vision, audio, depth, thermal) in a shared 1024-dim embedding space; `MultimodalEmbeddingRequest` accepts exactly one modality field; image/audio inputs written to named temp files (imagebind loaders require file paths); imagebind not on PyPI: `pip install git+https://github.com/facebookresearch/ImageBind.git` then `pip install 'sheaf-serve[multimodal]'`
- Cross-modal embeddings: ImageBind backend — five modalities (text, vision, audio, depth, thermal) in a shared 1024-dim embedding space; `MultimodalEmbeddingRequest` accepts exactly one modality field; image/audio inputs written to named temp files (imagebind loaders require file paths); imagebind not on PyPI: `pip install git+https://github.com/facebookresearch/ImageBind.git` then `pip install 'sheaf-serve[multimodal]'`
- Feast feature store: `FeatureRef` Pydantic model in `api/time_series.py`; `FeastResolver` in `integrations/feast.py` wraps `feast.FeatureStore`, resolves online features to `list[float]`; `feast_repo_path` field on `ModelSpec`; resolution happens per-request before batching in both `server.py` and `modal_server.py`; 502 on upstream Feast errors, 422 on missing `feast_repo_path`; smoke test in `test_smoke_feast.py`; example in `examples/quickstart_feast.py`; install with `pip install 'sheaf-serve[feast]'`
- Modal serverless: `ModalServer` in `modal_server.py` — zero-infra alternative to `ModelServer`; `backend_cls` modules cloudpickled by value via `register_pickle_by_value`; `AnyRequest` defined directly from lightweight API modules (no ray dep); `_build_asgi_app` shared ASGI builder; example in `examples/quickstart_modal.py`; install with `pip install 'sheaf-serve[modal]'`
- TabPFN integration test (gated on `TABPFN_TOKEN`): real `load()` + `fit()` + `predict()` against the live library; 8 tests in `test_tabpfn_integration.py`
- Ray Serve smoke coverage: all modalities have end-to-end smoke tests in `test_smoke_ray.py`
- Feast smoke coverage: real SQLite store, materialise → resolve → predict; 8 tests in `test_smoke_feast.py`; gated on `SHEAF_SMOKE_TEST=1`

**What works (v0.4):**
- Image diffusion: FLUX backend (`diffusers.FluxPipeline`) — FLUX.1-schnell (4 steps, guidance=0, Apache 2.0) and FLUX.1-dev (20-50 steps, guidance=3.5-7.0); `DiffusionRequest` → `DiffusionResponse` with base64-encoded PNG; bfloat16 by default; optional `enable_model_cpu_offload` for low-VRAM GPUs; seed returned in response for reproducibility; 17 mocked tests in `test_flux_backend.py`; install with `pip install 'sheaf-serve[diffusion]'`
- Video understanding: VideoMAE / TimeSformer backend (`transformers`) — any `AutoModel`-compatible video model; `VideoRequest` accepts base64-encoded frames; embedding task returns CLS or mean-pooled 768-dim (base) / 1024-dim (large) vectors; classification task returns top-5 softmax labels + scores over Kinetics-400; 17 mocked tests in `test_videomae_backend.py`; Ray Serve smoke tests cover both embedding and classification; install with `pip install 'sheaf-serve[video]'`

**What works (v0.5 Ops/DX — in progress):**
- Structured JSON logging: `sheaf.logging.JsonFormatter` + `configure_logging()`; gated by `SHEAF_LOG_JSON=1`; request_id / latency_ms / status in every predict log line; 15 tests in `test_logging.py`
- Prometheus metrics: `sheaf.metrics` module — `sheaf_requests_total`, `sheaf_request_duration_seconds`, `sheaf_batch_size_total`, `sheaf_backend_load_seconds`; `GET /metrics` per deployment; lazy import, `SHEAF_METRICS_DISABLED=1` guard; 337 tests passing; install with `pip install 'sheaf-serve[metrics]'`
- OpenTelemetry tracing: `sheaf.tracing` module — `sheaf.predict` span per request with sub-spans for Feast resolution (`sheaf.feast.resolve`) and backend inference (`sheaf.backend.infer`); `configure_tracing()` auto-configures SDK from `OTEL_EXPORTER_OTLP_ENDPOINT` or `SHEAF_OTEL_CONSOLE=1`; lazy import, `SHEAF_TRACING_DISABLED=1` guard, `_NoopTracer`/`_NoopSpan` shims when OTel absent; 24 tests in `test_tracing.py`; install with `pip install 'sheaf-serve[tracing]'`

## Repo layout

```
src/sheaf/
  __init__.py          # public exports: ModelServer, ModelSpec
  spec.py              # ModelSpec, ResourceConfig — declares what to serve
  server.py            # ModelServer + _SheafDeployment — Ray Serve orchestrator
  registry.py          # @register_backend decorator + _BACKEND_REGISTRY dict
  logging.py           # JsonFormatter + configure_logging — structured JSON logging (SHEAF_LOG_JSON=1)
  metrics.py           # Prometheus metrics: record_predict/batch/load, register_metrics_endpoint, time_load
  tracing.py           # OTel tracing: configure_tracing, get_tracer, trace_predict, trace_span, record_exception
  api/
    base.py            # BaseRequest, BaseResponse, ModelType enum
    time_series.py     # TimeSeriesRequest/Response, Frequency, OutputMode
    tabular.py         # TabularRequest/Response
    audio.py           # AudioRequest/Response, TTSRequest/TTSResponse
    audio_generation.py # AudioGenerationRequest/Response (MusicGen)
    embedding.py       # EmbeddingRequest/Response
    multimodal_embedding.py  # MultimodalEmbeddingRequest/Response (ImageBind)
    segmentation.py    # SegmentationRequest/Response
    depth.py           # DepthRequest/Response
    detection.py       # DetectionRequest/Response
    molecular.py       # MolecularRequest/Response (ESM-3)
    genomic.py         # GenomicRequest/Response (Nucleotide Transformer)
    small_molecule.py  # SmallMoleculeRequest/Response (MolFormer)
    materials.py       # MaterialsRequest/Response (MACE)
    satellite.py       # SatelliteRequest/Response (Prithvi)
    weather.py         # WeatherRequest/Response (GraphCast)
    diffusion.py       # DiffusionRequest/Response (FLUX)
    video.py           # VideoRequest/Response (VideoMAE / TimeSformer)
  backends/
    base.py            # ModelBackend ABC: load(), predict(), async_predict(), batch_predict()
    chronos.py         # Chronos2Backend — Chronos-Bolt + Chronos-T5 families
    moirai.py          # MoiraiBackend — Salesforce Moirai (uni2ts)
    tabpfn.py          # TabPFNBackend — TabPFN v2 classification + regression
    timesfm.py         # TimesFMBackend
    whisper.py         # WhisperBackend — openai-whisper (PyTorch)
    faster_whisper.py  # FasterWhisperBackend — faster-whisper (CTranslate2, no torch at runtime)
    bark.py            # BarkBackend — Bark TTS via HuggingFace transformers
    musicgen.py        # MusicGenBackend — MusicGen audio generation via HuggingFace transformers
    open_clip.py       # OpenCLIPBackend — image/text embeddings via open-clip-torch
    dinov2.py          # DINOv2Backend — image-only embeddings via HuggingFace transformers (CLS or mean pooling)
    sam2.py            # SAM2Backend — prompted image segmentation via sam2 library
    depth_anything.py  # DepthAnythingBackend — monocular depth estimation via transformers
    detr.py            # DETRBackend — object detection via DETR/RT-DETR (AutoModelForObjectDetection)
    esm3.py            # ESM3Backend — protein sequence embeddings via EvolutionaryScale esm (Python 3.12+)
    nucleotide_transformer.py  # NucleotideTransformerBackend — DNA/RNA embeddings via transformers
    molformer.py       # MolFormerBackend — SMILES embeddings via IBM MolFormer-XL (trust_remote_code=True)
    mace.py            # MACEBackend — MACE-MP-0 universal interatomic potential via ASE
    graphcast.py       # GraphCastBackend — weather forecasting via google-deepmind/graphcast (JAX/Haiku)
    prithvi.py         # PrithviBackend — IBM/NASA Prithvi-EO geospatial embeddings (trust_remote_code=True)
    imagebind.py       # ImageBindBackend — cross-modal embeddings (text/vision/audio/depth/thermal); imagebind not on PyPI
    flux.py            # FluxBackend — FLUX.1-schnell / FLUX.1-dev image diffusion via diffusers.FluxPipeline
    videomae.py        # VideoMAEBackend — video embeddings + classification via VideoMAE / TimeSformer
    _audio_utils.py    # Shared WAV encoding/decoding utility (no ffmpeg for WAV inputs)
  scheduling/
    batch.py           # BatchPolicy — wired into @serve.batch per deployment
  cache/
    __init__.py        # CacheConfig + ResponseCache — in-process LRU with optional TTL (SHEAF_CACHE_DISABLED=1)
  integrations/
    __init__.py        # exports FeastResolver
    feast.py           # FeastResolver — wraps feast.FeatureStore, resolves FeatureRef → list[float]
examples/
  quickstart.py        # Chronos time series example
  quickstart_tabular.py
  time_series_comparison.py  # Chronos vs TimesFM
  quickstart_audio.py        # Whisper + faster-whisper transcription, word timestamps, translation
  quickstart_vision.py       # DINOv2 + OpenCLIP image embeddings, CLS vs mean pooling, cross-modal retrieval
  quickstart_feast.py        # Feast feature store: build repo → materialise → feature_ref requests → Chronos forecasts
  quickstart_modal.py        # Modal serverless deployment with Chronos
  quickstart_diffusion.py    # FLUX.1-schnell: text-to-image generation, seed reproducibility, batch
  quickstart_diffusion_modal.py  # FLUX on Modal (T4): local-source workaround, tiny-flux-pipe for CI
  quickstart_video.py        # VideoMAE embeddings (CLS + mean pooling) + action classification
  quickstart_video_modal.py  # VideoMAE on Modal (T4): embed + classify synthetic clips
  quickstart_cache.py        # Request caching: CacheConfig on ModelSpec, timing first vs. cached call
  sample.wav                 # 4.8s 16kHz mono WAV for audio examples / smoke tests
tests/
  stubs.py             # Pytest-free stub backends for Ray worker cloudpickle
  test_api.py
  test_tabular_api.py
  test_server.py       # ModelBackend async dispatch, AnyRequest union, registry
  test_feast_resolver.py               # FeastResolver unit tests + _build_asgi_app Feast integration (12 tests)
  test_modal_server.py                 # ModalServer / _build_asgi_app tests (via TestClient)
  test_whisper_backend.py              # WhisperBackend mocked tests (8 tests)
  test_faster_whisper_backend.py       # FasterWhisperBackend mocked tests (9 tests)
  test_bark_backend.py                 # BarkBackend mocked tests (9 tests)
  test_musicgen_backend.py             # MusicGenBackend mocked tests (14 tests)
  test_open_clip_backend.py            # OpenCLIPBackend mocked tests (12 tests)
  test_dinov2_backend.py               # DINOv2Backend mocked tests (10 tests)
  test_sam2_backend.py                 # SAM2Backend mocked tests (11 tests)
  test_depth_anything_backend.py       # DepthAnythingBackend mocked tests (10 tests)
  test_detr_backend.py                 # DETRBackend mocked tests (11 tests)
  test_esm3_backend.py                 # ESM3Backend mocked tests (10 tests)
  test_nucleotide_transformer_backend.py  # NucleotideTransformerBackend mocked tests (13 tests)
  test_molformer_backend.py            # MolFormerBackend mocked tests (14 tests)
  test_mace_backend.py                 # MACEBackend mocked tests (13 tests)
  test_graphcast_backend.py            # GraphCastBackend mocked tests (16 tests)
  test_prithvi_backend.py              # PrithviBackend mocked tests (14 tests)
  test_imagebind_backend.py            # ImageBindBackend mocked tests (20 tests)
  test_flux_backend.py                 # FluxBackend mocked tests (17 tests)
  test_videomae_backend.py             # VideoMAEBackend mocked tests (17 tests)
  test_cache.py                        # CacheConfig, ResponseCache (LRU/TTL/key), and _build_asgi_app integration (27 tests)
  test_logging.py                      # JsonFormatter + configure_logging + server integration (15 tests)
  test_metrics.py                      # Prometheus metrics module (no-op absent/disabled + functional, 11+ tests)
  test_tracing.py                      # OTel tracing: NoopSpan/Tracer, configure_tracing, span attributes (24 tests)
  test_tabpfn_integration.py           # TabPFN integration tests — gated on TABPFN_TOKEN (8 tests)
  test_smoke_ray.py    # End-to-end Ray Serve tests (SHEAF_SMOKE_TEST=1 to run); covers all modalities
  test_smoke_whisper.py                # Whisper + faster-whisper e2e (SHEAF_SMOKE_TEST=1 to run)
  test_smoke_feast.py                  # Feast end-to-end: SQLite store, materialise, resolve, predict (SHEAF_SMOKE_TEST=1)
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
- **Request cache — opt-in per deployment** — `ModelSpec.cache = CacheConfig(enabled=True)` attaches an in-process LRU `ResponseCache` to that deployment. Default is `enabled=False` (no overhead). `SHEAF_CACHE_DISABLED=1` skips all caches process-wide regardless of spec config (useful in integration test runs where you want to exercise the backend every time).
- **Cache key design** — `ResponseCache.make_key(deployment, request)` serialises the request via `model_dump(mode="json", exclude={"request_id", ...})`, prepends the deployment name (so two specs with identical input types share no entries), JSON-canonicalises with `sort_keys=True`, then SHA-256 hashes. `request_id` is always excluded (it is unique per call). Additional fields can be excluded via `CacheConfig.exclude_fields` (e.g. `["seed"]` for diffusion models where different seeds should be distinct cache entries but you still want same-seed repeats to hit).
- **Cache placement — after Feast resolution** — the cache lookup and store happen after Feast feature resolution, inside the `trace_predict` span, before `_batch_predict`. This means the cache key reflects the actual history values (not a feature reference), so two requests for the same entity key at different times (with different resolved feature values) correctly produce distinct entries. The same placement applies in `modal_server.py`'s predict handler.
- **Cache store — dict, not response object** — `ResponseCache` stores `dict[str, Any]` (the `model_dump(mode="json")` output), not the Pydantic response object. This matches what the predict handler returns to FastAPI and avoids a second serialisation round-trip on cache hits.
- **`ResponseCache` is never cloudpickled** — `self._cache` is assigned in `_SheafDeployment.__init__`, which runs after `@serve.ingress` deserialises `_app` in the worker. Unlike `register_metrics_endpoint`, there is no cloudpickle hazard here.

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
- `bucket_by` batching: `BatchPolicy.bucket_by` field exists but grouping requests by horizon (or other field) before batching is not yet implemented
- `BatchPolicy` via `ModelServer.run()`: batch parameters are wired in `__init__` via setters; there is no separate `.options()` API for this in Ray Serve
