"""ModalServer — deploy Sheaf backends as a Modal ASGI web app.

Drop-in complement to ModelServer for teams that want Modal's managed
compute (GPU provisioning, autoscaling, zero-infra) instead of running
a Ray cluster themselves.

All specs are served by a single Modal ASGI function behind three routes
per model name — the same HTTP contract as ModelServer:

    GET  /{name}/health
    GET  /{name}/ready
    POST /{name}/predict

Usage (persistent deployment)::

    # my_server.py
    from sheaf import ModalServer
    from sheaf.spec import ModelSpec, ResourceConfig
    from sheaf.api.base import ModelType

    server = ModalServer(
        models=[chronos_spec, tabpfn_spec],
        app_name="my-sheaf",
        gpu="T4",
    )
    app = server.app   # expose at module scope so Modal CLI can find it

    # terminal:
    #   modal deploy my_server.py    # persistent deployment
    #   modal serve  my_server.py    # dev mode with hot reload

Notes:
    * All models share one container (same image and GPU).  If backends need
      incompatible deps, split them across separate ModalServer instances.
    * Modal charges per second of GPU compute.  min_containers=1 (default)
      keeps one warm instance alive; set to 0 to allow full scale-to-zero.
    * The ``app`` attribute is a ``modal.App`` — assign it to a module-level
      variable so ``modal deploy`` / ``modal serve`` can discover it.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

# Must be module-level: FastAPI resolves type hints via predict.__globals__
# (the module namespace), not the local scope of _build_asgi_app.
# Define AnyRequest here rather than importing from sheaf.server to avoid
# pulling in ray (and its heavy transitive deps) in Modal containers.
from sheaf.api.audio import AudioRequest, TTSRequest
from sheaf.api.audio_generation import AudioGenerationRequest
from sheaf.api.depth import DepthRequest
from sheaf.api.detection import DetectionRequest
from sheaf.api.diffusion import DiffusionRequest
from sheaf.api.embedding import EmbeddingRequest
from sheaf.api.genomic import GenomicRequest
from sheaf.api.materials import MaterialsRequest
from sheaf.api.molecular import MolecularRequest
from sheaf.api.multimodal_embedding import MultimodalEmbeddingRequest
from sheaf.api.satellite import SatelliteRequest
from sheaf.api.segmentation import SegmentationRequest
from sheaf.api.small_molecule import SmallMoleculeRequest
from sheaf.api.tabular import TabularRequest
from sheaf.api.time_series import TimeSeriesRequest
from sheaf.api.video import VideoRequest
from sheaf.api.weather import WeatherRequest
from sheaf.spec import ModelSpec

AnyRequest = Annotated[
    TimeSeriesRequest
    | TabularRequest
    | AudioRequest
    | AudioGenerationRequest
    | TTSRequest
    | EmbeddingRequest
    | SegmentationRequest
    | MolecularRequest
    | GenomicRequest
    | MaterialsRequest
    | SmallMoleculeRequest
    | DepthRequest
    | DetectionRequest
    | WeatherRequest
    | SatelliteRequest
    | MultimodalEmbeddingRequest
    | DiffusionRequest
    | VideoRequest,
    Field(discriminator="model_type"),
]

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ASGI app builder — runs *inside* the Modal container at startup
# ---------------------------------------------------------------------------


def _build_asgi_app(specs: list[ModelSpec]) -> Any:
    """Build a FastAPI ASGI app that serves all specs.

    Called once when the Modal container starts.  Loads every backend, then
    returns the FastAPI app that handles all requests for the container's
    lifetime.  ``specs`` is captured via closure and cloudpickled by Modal.
    """
    import importlib as _imp
    import os as _os

    from fastapi import FastAPI, HTTPException

    # Populate the backend registry (same pattern as _SheafDeployment.__init__).
    # Imports are lightweight — heavy deps stay lazy inside each backend's load().
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

    for _mod in _os.environ.get("SHEAF_EXTRA_BACKENDS", "").split(","):
        if _mod.strip():
            _imp.import_module(_mod.strip())

    from sheaf.registry import _BACKEND_REGISTRY

    # Load all backends (and optional Feast resolvers) once — they stay in
    # memory for the container's lifetime.
    # _backends maps name → (spec, backend, feast_resolver | None)
    _backends: dict[str, tuple[ModelSpec, Any, Any]] = {}
    for spec in specs:
        backend_cls = spec.backend_cls or _BACKEND_REGISTRY.get(spec.backend)
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend '{spec.backend}'. "
                f"Registered backends: {list(_BACKEND_REGISTRY)}"
            )
        backend = backend_cls(**spec.backend_kwargs)
        backend.load()

        feast = None
        if spec.feast_repo_path:
            from sheaf.integrations.feast import FeastResolver

            feast = FeastResolver(spec.feast_repo_path)
            feast.load()
            _logger.info(
                "Feast resolver loaded for '%s' (repo: %s)",
                spec.name,
                spec.feast_repo_path,
            )

        _backends[spec.name] = (spec, backend, feast)
        _logger.info("Loaded backend '%s' (%s)", spec.name, spec.backend)

    asgi_app = FastAPI(title="Sheaf")

    @asgi_app.get("/{name}/health")
    def health(name: str) -> dict[str, str]:
        if name not in _backends:
            raise HTTPException(status_code=404, detail=f"No deployment named '{name}'")
        return {"status": "ok"}

    @asgi_app.get("/{name}/ready")
    def ready(name: str) -> dict[str, str]:
        if name not in _backends:
            raise HTTPException(status_code=404, detail=f"No deployment named '{name}'")
        spec, _, _feast = _backends[name]
        return {"status": "ready", "model": spec.name}

    @asgi_app.post("/{name}/predict")
    async def predict(name: str, request: AnyRequest) -> dict[str, Any]:  # type: ignore[valid-type]
        if name not in _backends:
            raise HTTPException(status_code=404, detail=f"No deployment named '{name}'")
        spec, backend, feast = _backends[name]
        if request.model_type != spec.model_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Backend '{name}' expects model_type='{spec.model_type}', "
                    f"got '{request.model_type}'"
                ),
            )

        # Feast feature resolution — per-request, before inference.
        _feature_ref = getattr(request, "feature_ref", None)
        if _feature_ref is not None:
            if feast is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Request contains feature_ref but ModelSpec '{name}' "
                        "has no feast_repo_path configured."
                    ),
                )
            try:
                history = feast.resolve(_feature_ref)
            except Exception as exc:
                _logger.exception("Feast resolution error in deployment '%s'", name)
                raise HTTPException(
                    status_code=502,
                    detail=f"Feast resolution failed: {type(exc).__name__}: {exc}",
                ) from exc
            request = request.model_copy(
                update={"history": history, "feature_ref": None}
            )

        try:
            result = await backend.async_predict(request)
            return result.model_dump(mode="json")
        except Exception as exc:
            _logger.exception("Inference error in deployment '%s'", name)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    return asgi_app


# ---------------------------------------------------------------------------
# ModalServer
# ---------------------------------------------------------------------------


class ModalServer:
    """Deploy Sheaf backends as a Modal ASGI web app.

    Mirrors the ``ModelServer`` interface but targets Modal instead of Ray Serve.
    All specs are served under ``/{name}/health``, ``/{name}/ready``, and
    ``/{name}/predict`` — the same HTTP contract as ``ModelServer``.

    All models run in a single Modal container.  If backends need incompatible
    dependencies (e.g. moirai vs. vision), split them into separate
    ``ModalServer`` instances each with a tailored ``image``.

    Example::

        # my_server.py
        from sheaf import ModalServer
        from sheaf.spec import ModelSpec, ResourceConfig
        from sheaf.api.base import ModelType

        server = ModalServer(
            models=[chronos_spec, tabpfn_spec],
            app_name="my-sheaf",
            gpu="T4",
        )
        app = server.app   # <-- module-scope exposure for Modal CLI

        # terminal:
        #   modal deploy my_server.py    # persistent
        #   modal serve  my_server.py    # dev / hot-reload

    Args:
        models: List of ``ModelSpec`` — same as ``ModelServer``.
        app_name: Name of the Modal app (shown in the Modal dashboard).
        gpu: Modal GPU type, e.g. ``"T4"``, ``"A10G"``, ``"H100"``.
            ``None`` for CPU-only containers.
        image: ``modal.Image`` to use.  Defaults to ``debian_slim`` with
            ``sheaf-serve`` installed.  Pass a custom image to add backend
            extras, e.g.::

                modal.Image.debian_slim().pip_install("sheaf-serve[time-series]")

        min_containers: Minimum number of warm containers (default 1 to avoid
            cold starts during demos).  Set to 0 for full scale-to-zero.
    """

    def __init__(
        self,
        models: list[ModelSpec],
        app_name: str = "sheaf",
        gpu: str | None = None,
        image: Any = None,
        min_containers: int = 1,
    ) -> None:
        try:
            import modal
        except ImportError as e:
            raise ImportError(
                "modal is required for ModalServer. Install it with: pip install modal"
            ) from e

        self._models = models

        _image = image or (
            modal.Image.debian_slim(python_version="3.11").pip_install("sheaf-serve")
        )

        # Pre-pickle specs with cloudpickle so that any backend_cls classes
        # defined in the caller's module (not importable in the remote container)
        # are embedded by value rather than referenced by module path.
        # Ray (a sheaf-serve dep) always installs cloudpickle, so it is
        # available both locally and in the container image.
        import sys

        import cloudpickle as _cloudpickle

        # Register each backend_cls's defining module for by-value pickling.
        # Without this, cloudpickle serializes classes by module path + name,
        # which fails in the container because the user's script is never there.
        for _spec in models:
            if _spec.backend_cls is not None:
                _cls_mod = _spec.backend_cls.__module__
                if _cls_mod in sys.modules:
                    _cloudpickle.register_pickle_by_value(sys.modules[_cls_mod])

        _specs_bytes: bytes = _cloudpickle.dumps(models)

        modal_app = modal.App(app_name)

        @modal_app.function(
            image=_image,
            gpu=gpu,
            min_containers=min_containers,
            serialized=True,  # closure — cloudpickle instead of import-by-name
        )
        @modal.asgi_app()
        def _serve() -> Any:
            import cloudpickle as _cp

            _specs = _cp.loads(_specs_bytes)
            return _build_asgi_app(_specs)

        #: ``modal.App`` instance.  Assign to a module-level variable so the
        #: Modal CLI can discover it::
        #:
        #:     app = server.app   # module scope
        #:     # modal deploy my_server.py
        self.app: Any = modal_app
        self._serve_fn = _serve
