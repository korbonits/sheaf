"""ModelSpec — declares what Sheaf should serve."""

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

    Example:
        spec = ModelSpec(
            name="chronos2-small",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={"model_size": "small"},
            resources=ResourceConfig(num_gpus=1, replicas=2),
            batch_policy=BatchPolicy(max_batch_size=64, bucket_by="horizon"),
        )
    """

    name: str
    model_type: ModelType
    backend: str
    backend_kwargs: dict = Field(default_factory=dict)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    batch_policy: BatchPolicy = Field(default_factory=BatchPolicy)
