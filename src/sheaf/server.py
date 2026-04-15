"""ModelServer — the Ray Serve entry point for Sheaf."""

from typing import Any, cast

import ray
from ray import serve

from sheaf.api.base import BaseRequest, BaseResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import _BACKEND_REGISTRY, register_backend  # noqa: F401
from sheaf.spec import ModelSpec


@serve.deployment
class _SheafDeployment:
    def __init__(self, spec: ModelSpec) -> None:
        backend_cls = _BACKEND_REGISTRY.get(spec.backend)
        if backend_cls is None:
            raise ValueError(
                f"Unknown backend '{spec.backend}'. "
                f"Registered backends: {list(_BACKEND_REGISTRY)}"
            )
        self._backend: ModelBackend = backend_cls(**spec.backend_kwargs)
        self._backend.load()
        self._spec = spec

    async def __call__(self, request: BaseRequest) -> BaseResponse:
        return self._backend.predict(request)


class ModelServer:
    """Top-level serving orchestrator.

    Example:
        server = ModelServer(
            models=[chronos_spec, tabpfn_spec],
        )
        server.run()
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

    def run(self) -> None:
        if not ray.is_initialized():
            ray.init()

        serve.start(http_options={"host": self._host, "port": self._port})

        for spec in self._models:
            # Ray Serve adds .options() dynamically via @serve.deployment decorator
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
            handle = serve.run(deployment, name=spec.name, route_prefix=f"/{spec.name}")
            self._deployments[spec.name] = handle

    def shutdown(self) -> None:
        serve.shutdown()
