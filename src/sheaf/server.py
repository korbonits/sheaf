"""ModelServer — the Ray Serve entry point for Sheaf."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Annotated, Any, cast

import ray
from fastapi import FastAPI, HTTPException
from pydantic import Field
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
import sheaf.backends.moirai  # noqa: F401
import sheaf.backends.open_clip  # noqa: F401
import sheaf.backends.sam2  # noqa: F401
import sheaf.backends.tabpfn  # noqa: F401
import sheaf.backends.timesfm  # noqa: F401
import sheaf.backends.whisper  # noqa: F401
from sheaf.api.audio import AudioRequest, TTSRequest
from sheaf.api.base import BaseRequest
from sheaf.api.depth import DepthRequest
from sheaf.api.detection import DetectionRequest
from sheaf.api.embedding import EmbeddingRequest
from sheaf.api.molecular import MolecularRequest
from sheaf.api.segmentation import SegmentationRequest
from sheaf.api.tabular import TabularRequest
from sheaf.api.time_series import TimeSeriesRequest
from sheaf.backends.base import ModelBackend
from sheaf.registry import _BACKEND_REGISTRY, register_backend  # noqa: F401
from sheaf.spec import ModelSpec

# Allow extra backend modules to be registered at worker startup via an env var.
# Useful for custom backends and testing:
#   SHEAF_EXTRA_BACKENDS=mypackage.backends,tests.stubs
for _mod in os.environ.get("SHEAF_EXTRA_BACKENDS", "").split(","):
    if _mod.strip():
        importlib.import_module(_mod.strip())

# Discriminated union of all supported request types.
# FastAPI uses the `model_type` field to select the right Pydantic model
# and return 422 if the body doesn't match any variant.
AnyRequest = Annotated[
    TimeSeriesRequest
    | TabularRequest
    | AudioRequest
    | TTSRequest
    | EmbeddingRequest
    | SegmentationRequest
    | MolecularRequest
    | DepthRequest
    | DetectionRequest,
    Field(discriminator="model_type"),
]

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
        import sheaf.backends.moirai  # noqa: F401
        import sheaf.backends.open_clip  # noqa: F401
        import sheaf.backends.sam2  # noqa: F401
        import sheaf.backends.tabpfn  # noqa: F401
        import sheaf.backends.timesfm  # noqa: F401
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
        self._backend: ModelBackend = backend_cls(**spec.backend_kwargs)
        self._backend.load()
        self._spec = spec

        # Wire BatchPolicy into the @serve.batch handler.
        # set_max_batch_size / set_batch_wait_timeout_s are the official Ray
        # Serve API for runtime batch parameter updates (see serve.batch docs).
        self._batch_predict.set_max_batch_size(  # ty: ignore[unresolved-attribute]
            spec.batch_policy.max_batch_size
        )
        self._batch_predict.set_batch_wait_timeout_s(  # ty: ignore[unresolved-attribute]
            spec.batch_policy.timeout_ms / 1000.0
        )

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
        try:
            return await self._batch_predict(request)  # type: ignore[return-value]
        except Exception as exc:
            # Log the full traceback server-side; expose only type + message
            # to the client so internal details don't leak.
            _logger.exception("Inference error in deployment '%s'", self._spec.name)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def _batch_predict(self, requests: list[BaseRequest]) -> list[dict[str, Any]]:
        """Batched inference handler.

        Ray Serve accumulates concurrent predict() calls and delivers them
        here as a list.  Returns one dict per request in the same order.

        The decorator defaults (32 / 50 ms) are overridden per deployment by
        __init__ using the ModelSpec.batch_policy values.
        """
        responses = await self._backend.async_batch_predict(requests)
        return [r.model_dump(mode="json") for r in responses]


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
