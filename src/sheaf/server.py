"""ModelServer — the Ray Serve entry point for Sheaf."""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from typing import Any, cast

import ray
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from ray import serve

# Auto-import standard backends so they self-register via @register_backend.
# These are side-effect-only imports; heavy deps (torch, etc.) are lazy inside load().
# This ensures the registry is populated in Ray Serve worker processes.
import sheaf.backends.bark  # noqa: F401
import sheaf.backends.chronos  # noqa: F401
import sheaf.backends.depth_anything  # noqa: F401
import sheaf.backends.detr  # noqa: F401
import sheaf.backends.dinov2  # noqa: F401
import sheaf.backends.esm3  # noqa: F401
import sheaf.backends.faster_whisper  # noqa: F401
import sheaf.backends.flux  # noqa: F401
import sheaf.backends.graphcast  # noqa: F401
import sheaf.backends.imagebind  # noqa: F401
import sheaf.backends.kokoro  # noqa: F401
import sheaf.backends.mace  # noqa: F401
import sheaf.backends.moirai  # noqa: F401
import sheaf.backends.molformer  # noqa: F401
import sheaf.backends.musicgen  # noqa: F401
import sheaf.backends.nucleotide_transformer  # noqa: F401
import sheaf.backends.open_clip  # noqa: F401
import sheaf.backends.pointnet  # noqa: F401
import sheaf.backends.prithvi  # noqa: F401
import sheaf.backends.raft  # noqa: F401
import sheaf.backends.sam2  # noqa: F401
import sheaf.backends.sdxl  # noqa: F401
import sheaf.backends.tabpfn  # noqa: F401
import sheaf.backends.timesfm  # noqa: F401
import sheaf.backends.videomae  # noqa: F401
import sheaf.backends.vitpose  # noqa: F401
import sheaf.backends.whisper  # noqa: F401
from sheaf.api.base import BaseRequest
from sheaf.api.union import AnyRequest
from sheaf.backends.base import ModelBackend
from sheaf.cache import _DISABLED as _CACHE_DISABLED
from sheaf.cache import ResponseCache
from sheaf.lora import bucket_with_adapter_resolution, resolve_active_adapters
from sheaf.metrics import (
    record_batch,
    record_predict,
    register_metrics_endpoint,
    time_load,
)
from sheaf.registry import _BACKEND_REGISTRY, register_backend  # noqa: F401
from sheaf.scheduling.batch import bucket_requests
from sheaf.spec import ModelSpec
from sheaf.tracing import configure_tracing, record_exception, trace_predict, trace_span

# Allow extra backend modules to be registered at worker startup via an env var.
# Useful for custom backends and testing:
#   SHEAF_EXTRA_BACKENDS=mypackage.backends,tests.stubs
for _mod in os.environ.get("SHEAF_EXTRA_BACKENDS", "").split(","):
    if _mod.strip():
        importlib.import_module(_mod.strip())

_app = FastAPI(title="Sheaf")
_logger = logging.getLogger(__name__)


@serve.deployment
@serve.ingress(_app)
class _SheafDeployment:
    def __init__(self, spec: ModelSpec) -> None:
        # Ray Serve may cloudpickle this class by value (inline), causing two
        # problems in the worker process:
        #
        #   1. sheaf.server is never imported → module-level SHEAF_EXTRA_BACKENDS
        #      loop never runs.
        #   2. The captured _BACKEND_REGISTRY reference is a snapshot dict from
        #      the driver process, not the live worker registry.
        #
        # Additionally, even when the class IS imported by module reference,
        # the module-level loop runs before runtime_env.env_vars are applied,
        # so SHEAF_EXTRA_BACKENDS is still empty at that point.
        #
        # Fix: re-import standard backends (idempotent), run the extra-backends
        # loop here (guaranteed to see runtime_env env_vars), then look up from
        # the freshly-populated worker registry.
        import sheaf.backends.bark  # noqa: F401
        import sheaf.backends.chronos  # noqa: F401
        import sheaf.backends.depth_anything  # noqa: F401
        import sheaf.backends.detr  # noqa: F401
        import sheaf.backends.dinov2  # noqa: F401
        import sheaf.backends.esm3  # noqa: F401
        import sheaf.backends.faster_whisper  # noqa: F401
        import sheaf.backends.flux  # noqa: F401
        import sheaf.backends.graphcast  # noqa: F401
        import sheaf.backends.imagebind  # noqa: F401
        import sheaf.backends.mace  # noqa: F401
        import sheaf.backends.moirai  # noqa: F401
        import sheaf.backends.molformer  # noqa: F401
        import sheaf.backends.musicgen  # noqa: F401
        import sheaf.backends.nucleotide_transformer  # noqa: F401
        import sheaf.backends.open_clip  # noqa: F401
        import sheaf.backends.prithvi  # noqa: F401
        import sheaf.backends.sam2  # noqa: F401
        import sheaf.backends.tabpfn  # noqa: F401
        import sheaf.backends.timesfm  # noqa: F401
        import sheaf.backends.videomae  # noqa: F401
        import sheaf.backends.whisper  # noqa: F401

        for _mod in os.environ.get("SHEAF_EXTRA_BACKENDS", "").split(","):
            if _mod.strip():
                importlib.import_module(_mod.strip())

        # Import the live registry — not the module-level binding which may be
        # a stale cloudpickle snapshot.
        from sheaf.registry import _BACKEND_REGISTRY as _registry

        backend_cls = spec.backend_cls or _registry.get(spec.backend)
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend '{spec.backend}'. "
                f"Registered backends: {list(_registry)}"
            )
        if os.environ.get("SHEAF_LOG_JSON"):
            from sheaf.logging import configure_logging

            configure_logging()

        configure_tracing()

        self._backend: ModelBackend = backend_cls(**spec.backend_kwargs)
        with time_load(spec.name, spec.model_type):
            self._backend.load()
        self._spec = spec
        _logger.info(
            "backend loaded",
            extra={
                "deployment": spec.name,
                "backend": spec.backend or type(self._backend).__name__,
                "model_type": spec.model_type,
            },
        )

        # LoRA adapter loading — backend must opt in via supports_lora().
        if spec.lora is not None:
            if not self._backend.supports_lora():
                raise ValueError(
                    f"ModelSpec '{spec.name}' configured with LoRA adapters "
                    f"but backend {type(self._backend).__name__} does not "
                    f"support them (supports_lora() returned False)."
                )
            self._backend.load_adapters(spec.lora.adapters)
            _logger.info(
                "LoRA adapters loaded",
                extra={
                    "deployment": spec.name,
                    "adapters": list(spec.lora.adapters),
                    "default": spec.lora.default,
                },
            )

        # Feast resolver — only initialised when the spec provides a repo path.
        self._feast: Any = None
        if spec.feast_repo_path:
            from sheaf.integrations.feast import FeastResolver

            self._feast = FeastResolver(spec.feast_repo_path)
            self._feast.load()
            _logger.info(
                "Feast resolver loaded for '%s' (repo: %s)",
                spec.name,
                spec.feast_repo_path,
            )

        # Response cache — disabled globally if SHEAF_CACHE_DISABLED=1.
        self._cache: ResponseCache | None = None
        if not _CACHE_DISABLED and spec.cache.enabled:
            self._cache = ResponseCache(spec.cache)

        # Wire BatchPolicy into the @serve.batch handler.
        # set_max_batch_size / set_batch_wait_timeout_s are the official Ray
        # Serve API for runtime batch parameter updates (see serve.batch docs).
        self._batch_predict.set_max_batch_size(  # ty: ignore[unresolved-attribute]
            spec.batch_policy.max_batch_size
        )
        self._batch_predict.set_batch_wait_timeout_s(  # ty: ignore[unresolved-attribute]
            spec.batch_policy.timeout_ms / 1000.0
        )

        # Prometheus /metrics endpoint — registered here (not at module level)
        # because @serve.ingress cloudpickles _app at class-definition time.
        # Nested functions whose __globals__ include prometheus_client internals
        # (CollectorRegistry has threading.Lock) cause RecursionError during
        # that serialisation.  Running here is safe: _app is already
        # deserialised in the worker process and never re-pickled.
        register_metrics_endpoint(_app, spec.name)

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------

    @_app.get("/health")
    async def health(self) -> dict[str, str]:
        """Liveness probe — returns 200 as long as the process is up."""
        return {"status": "ok"}

    @_app.get("/ready")
    async def ready(self) -> dict[str, str]:
        """Readiness probe — returns 200 once the model is loaded."""
        return {"status": "ready", "model": self._spec.name}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @_app.post("/predict")
    async def predict(self, request: AnyRequest) -> dict[str, Any]:
        """Single-item predict.

        Requests are automatically batched by the @serve.batch handler
        below — concurrent calls are grouped up to max_batch_size or
        batch_wait_timeout_s, whichever comes first.

        If the request carries a ``feature_ref``, the Feast resolver fetches
        the online feature and replaces ``feature_ref`` with the resolved
        ``history`` before the request reaches the batching layer.  Resolution
        is per-request (not per-batch) because each request may reference a
        different entity.
        """
        if request.model_type != self._spec.model_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Backend '{self._spec.name}' expects "
                    f"model_type='{self._spec.model_type}', "
                    f"got '{request.model_type}'"
                ),
            )

        _request_adapters = getattr(request, "adapters", None) or []
        if _request_adapters:
            if self._spec.lora is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Request specifies adapters {_request_adapters} but "
                        f"deployment '{self._spec.name}' has no LoRA configured."
                    ),
                )
            _unknown = [
                n for n in _request_adapters if n not in self._spec.lora.adapters
            ]
            if _unknown:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Unknown adapter(s) {_unknown}; deployment "
                        f"'{self._spec.name}' has: {sorted(self._spec.lora.adapters)}"
                    ),
                )

        with trace_predict(
            self._spec.name,
            str(request.model_type),
            request.model_name or "",
            str(request.request_id),
        ) as _span:
            # Feast feature resolution — runs before batching so each entity
            # is looked up independently and the backend always sees `history`.
            _feature_ref = getattr(request, "feature_ref", None)
            if _feature_ref is not None:
                if self._feast is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Request contains feature_ref but ModelSpec "
                            f"'{self._spec.name}' has no feast_repo_path configured."
                        ),
                    )
                try:
                    with trace_span("sheaf.feast.resolve", deployment=self._spec.name):
                        history = self._feast.resolve(_feature_ref)
                except Exception as exc:
                    _logger.exception(
                        "Feast resolution error in deployment '%s'", self._spec.name
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"Feast resolution failed: {type(exc).__name__}: {exc}",
                    ) from exc
                request = request.model_copy(
                    update={"history": history, "feature_ref": None}
                )

            # Cache lookup — after Feast resolution so key reflects actual input.
            _cache_key: str | None = None
            if self._cache is not None:
                _cache_key = self._cache.make_key(self._spec.name, request)
                _cached = self._cache.get(_cache_key)
                if _cached is not None:
                    return _cached

            _t0 = time.perf_counter()
            _log_extra: dict[str, object] = {
                "request_id": str(request.request_id),
                "deployment": self._spec.name,
                "model_type": request.model_type,
                "model_name": request.model_name,
            }
            try:
                with trace_span("sheaf.backend.infer", deployment=self._spec.name):
                    result = await self._batch_predict(request)  # type: ignore[return-value]
                _latency_s = time.perf_counter() - _t0
                _log_extra["latency_ms"] = round(_latency_s * 1000, 2)
                _log_extra["status"] = "ok"
                _logger.info("predict ok", extra=_log_extra)
                record_predict(self._spec.name, request.model_type, "ok", _latency_s)
                if self._cache is not None and _cache_key is not None:
                    self._cache.set(_cache_key, result)
                return result
            except Exception as exc:
                _latency_s = time.perf_counter() - _t0
                _log_extra["latency_ms"] = round(_latency_s * 1000, 2)
                _log_extra["status"] = "error"
                _log_extra["error"] = f"{type(exc).__name__}: {exc}"
                _logger.exception("predict error", extra=_log_extra)
                record_predict(self._spec.name, request.model_type, "error", _latency_s)
                record_exception(_span, exc)
                raise HTTPException(
                    status_code=500,
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc

    @_app.post("/stream")
    async def stream(self, request: AnyRequest) -> StreamingResponse:
        """Stream inference events via Server-Sent Events (SSE).

        Returns a ``text/event-stream`` response.  Each event is a JSON object
        on a ``data: <json>\\n\\n`` line.  Two event shapes:

        - Progress: ``{"type": "progress", "step": N, "total_steps": N, "done": false}``
        - Result:   ``{"type": "result", "done": true, ...response_fields}``

        Bypasses batching and the response cache — each stream is per-request.
        Feast feature resolution runs identically to ``/predict``.
        """
        if request.model_type != self._spec.model_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Backend '{self._spec.name}' expects "
                    f"model_type='{self._spec.model_type}', "
                    f"got '{request.model_type}'"
                ),
            )

        _stream_adapters = getattr(request, "adapters", None) or []
        if _stream_adapters:
            if self._spec.lora is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Request specifies adapters {_stream_adapters} but "
                        f"deployment '{self._spec.name}' has no LoRA configured."
                    ),
                )
            _unknown = [
                n for n in _stream_adapters if n not in self._spec.lora.adapters
            ]
            if _unknown:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Unknown adapter(s) {_unknown}; deployment "
                        f"'{self._spec.name}' has: {sorted(self._spec.lora.adapters)}"
                    ),
                )
            # Stream activates the resolved adapters before backend.stream_predict.
            _resolved_names, _resolved_weights = resolve_active_adapters(
                request, self._spec.lora
            )
            self._backend.set_active_adapters(_resolved_names, _resolved_weights)
        elif self._spec.lora is not None and self._spec.lora.default is not None:
            _resolved_names, _resolved_weights = resolve_active_adapters(
                request, self._spec.lora
            )
            self._backend.set_active_adapters(_resolved_names, _resolved_weights)

        _feature_ref = getattr(request, "feature_ref", None)
        if _feature_ref is not None:
            if self._feast is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Request contains feature_ref but ModelSpec "
                        f"'{self._spec.name}' has no feast_repo_path configured."
                    ),
                )
            try:
                history = self._feast.resolve(_feature_ref)
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Feast resolution failed: {type(exc).__name__}: {exc}",
                ) from exc
            request = request.model_copy(
                update={"history": history, "feature_ref": None}
            )

        async def _sse() -> Any:
            try:
                async for event in self._backend.stream_predict(request):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def _batch_predict(self, requests: list[BaseRequest]) -> list[dict[str, Any]]:
        """Batched inference handler.

        Ray Serve accumulates concurrent predict() calls and delivers them
        here as a list.  Returns one dict per request in the same order.

        The decorator defaults (32 / 50 ms) are overridden per deployment by
        __init__ using the ModelSpec.batch_policy values.

        Two grouping modes:

          - When ``batch_policy.bucket_by`` is set, requests are grouped by
            the value of that field before being dispatched to the backend,
            so each ``async_batch_predict`` call receives a homogeneous
            sub-batch (e.g. time-series ``horizon`` or video ``n_frames``).
          - When ``ModelSpec.lora`` is set, requests are grouped by their
            *resolved* (names, weights) adapter selection.  ``set_active_adapters``
            is called once per group before dispatch, since
            ``pipeline.set_adapters`` is process-global state that cannot be
            shared across concurrent requests with different adapters.

        These two modes are mutually exclusive (enforced by ``ModelSpec``).
        Results are reassembled in the original arrival order before return.
        """
        record_batch(self._spec.name, len(requests))

        if self._spec.lora is not None:
            lora_groups = bucket_with_adapter_resolution(requests, self._spec.lora)
            if (
                len(lora_groups) == 1
                and not lora_groups[0][2]  # no active adapters anywhere
            ):
                responses = await self._backend.async_batch_predict(lora_groups[0][1])
                return [r.model_dump(mode="json") for r in responses]
            slot: dict[int, dict[str, Any]] = {}
            for indices, sub_reqs, names, weights in lora_groups:
                self._backend.set_active_adapters(names, weights)
                bucket_responses = await self._backend.async_batch_predict(sub_reqs)
                for idx, resp in zip(indices, bucket_responses):
                    slot[idx] = resp.model_dump(mode="json")
            return [slot[i] for i in range(len(requests))]

        groups = bucket_requests(requests, self._spec.batch_policy.bucket_by)
        if len(groups) == 1:
            # Common path: no bucketing, or all requests share the same value.
            responses = await self._backend.async_batch_predict(groups[0][1])
            return [r.model_dump(mode="json") for r in responses]

        # Multiple buckets — dispatch each independently, reassemble in order.
        slot = {}
        for indices, sub_reqs in groups:
            bucket_responses = await self._backend.async_batch_predict(sub_reqs)
            for idx, resp in zip(indices, bucket_responses):
                slot[idx] = resp.model_dump(mode="json")
        return [slot[i] for i in range(len(requests))]


class ModelServer:
    """Top-level serving orchestrator.

    Example:
        server = ModelServer(
            models=[chronos_spec, tabpfn_spec],
        )
        server.run()

    Each model is deployed at /<name>/predict, /<name>/health,
    and /<name>/ready.
    """

    def __init__(
        self,
        models: list[ModelSpec],
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._models = models
        self._host = host
        self._port = port
        self._deployments: dict[str, Any] = {}

    def _deploy(self, spec: ModelSpec) -> Any:
        """Build and deploy a _SheafDeployment for spec. Returns the handle."""
        deployment = (
            cast(Any, _SheafDeployment)
            .options(
                name=spec.name,
                num_replicas=spec.resources.replicas,
                ray_actor_options={
                    "num_cpus": spec.resources.num_cpus,
                    "num_gpus": spec.resources.num_gpus,
                },
            )
            .bind(spec)
        )
        return serve.run(deployment, name=spec.name, route_prefix=f"/{spec.name}")

    def run(self) -> None:
        if os.environ.get("SHEAF_LOG_JSON"):
            from sheaf.logging import configure_logging

            configure_logging()

        configure_tracing()

        if not ray.is_initialized():
            ray.init()

        serve.start(http_options={"host": self._host, "port": self._port})

        for spec in self._models:
            self._deployments[spec.name] = self._deploy(spec)

    def update(self, spec: ModelSpec) -> None:
        """Hot-swap a running deployment with a new ModelSpec.

        Ray Serve performs a rolling update: new replicas start with the new
        spec while old replicas finish in-flight requests. The route prefix
        and deployment name are preserved, so clients see no URL change and
        no requests are dropped.

        The deployment must already be running (spec.name was passed to run()).
        To add a brand-new deployment, call run() again with an updated models
        list instead.

        Example:
            server.update(ModelSpec(
                name="chronos-small",          # same name — replaces in place
                backend="chronos2",
                backend_kwargs={"model_size": "large"},  # new weights
                resources=ResourceConfig(num_gpus=1),
            ))
        """
        if spec.name not in self._deployments:
            raise ValueError(
                f"No deployment named '{spec.name}'. "
                f"Running deployments: {list(self._deployments)}"
            )
        self._deployments[spec.name] = self._deploy(spec)
        # Keep self._models in sync so repeated calls to run() are consistent.
        self._models = [s if s.name != spec.name else spec for s in self._models]

    def shutdown(self) -> None:
        serve.shutdown()
