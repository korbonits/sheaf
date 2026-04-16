"""ModelSpec — declares what Sheaf should serve."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sheaf.api.base import ModelType
from sheaf.scheduling.batch import BatchPolicy


class ResourceConfig(BaseModel):
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory_gb: float | None = None
    replicas: int = 1


class ModelSpec(BaseModel):
    """Declares a model to be served by Sheaf.

    Example (registry-based — backend module must be imported first):
        spec = ModelSpec(
            name="chronos2-small",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={"model_size": "small"},
            resources=ResourceConfig(num_gpus=1, replicas=2),
            batch_policy=BatchPolicy(max_batch_size=64, bucket_by="horizon"),
        )

    Example (class-based — useful for custom or test backends):
        spec = ModelSpec(
            name="my-model",
            model_type=ModelType.TIME_SERIES,
            backend="my-model",
            backend_cls=MyModelBackend,
        )

    When backend_cls is provided it takes precedence over the registry lookup.
    Ray Serve serializes the class via cloudpickle, so it works for locally-
    defined backends without needing a separate module import in the worker.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    model_type: ModelType
    backend: str
    # type[ModelBackend] | None — Any avoids Pydantic schema generation issues
    backend_cls: Any = None
    backend_kwargs: dict = Field(default_factory=dict)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    batch_policy: BatchPolicy = Field(default_factory=BatchPolicy)
    feast_repo_path: str | None = Field(
        default=None,
        description=(
            "Path to a Feast feature repo directory (contains feature_store.yaml). "
            "When set, TimeSeriesRequests that specify feature_ref will have their "
            "history resolved from the online store before inference."
        ),
    )
