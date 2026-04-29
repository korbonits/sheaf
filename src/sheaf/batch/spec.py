"""BatchSpec — declares what Sheaf should run as an offline batch job.

Mirrors ``ModelSpec`` for the backend-selection fields (``name``,
``model_type``, ``backend``, ``backend_cls``, ``backend_kwargs``) and adds
``source`` + ``sink`` + execution config for offline pipelines.

v0.6 ships only JSONL I/O.  S3 / Parquet / Delta sources and sinks are
tracked in follow-up issues and will slot in as additional
``BatchSource``/``BatchSink`` subclasses without changing the runner API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sheaf.api.base import ModelType


class BatchSource(BaseModel):
    """Base class for batch input sources.  Subclass to add new formats."""

    model_config = {"extra": "forbid"}


class BatchSink(BaseModel):
    """Base class for batch output sinks.  Subclass to add new formats."""

    model_config = {"extra": "forbid"}


class JsonlSource(BatchSource):
    """JSONL file input — one request dict per line."""

    path: str


class JsonlSink(BatchSink):
    """JSONL file output — one response dict per line.

    The file is overwritten if it already exists.  Output rows are written
    in the same order as input rows.
    """

    path: str


class BatchSpec(BaseModel):
    """Declares an offline batch inference job.

    Example:
        spec = BatchSpec(
            name="chronos-batch",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={"model_size": "small"},
            source=JsonlSource(path="inputs.jsonl"),
            sink=JsonlSink(path="outputs.jsonl"),
            batch_size=64,
        )
        BatchRunner(spec).run()

    Fields ``backend`` / ``backend_cls`` / ``backend_kwargs`` follow the
    same semantics as on ``ModelSpec``: ``backend_cls`` takes precedence
    over the registry lookup when set, letting callers pass a custom or
    test-only backend class directly.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    model_type: ModelType
    backend: str
    # type[ModelBackend] | None — Any avoids Pydantic schema generation issues
    backend_cls: Any = None
    backend_kwargs: dict = Field(default_factory=dict)
    source: BatchSource
    sink: BatchSink
    batch_size: int = Field(
        default=32,
        gt=0,
        description=(
            "Rows per backend.batch_predict() call.  Controls the inner "
            "batch size Ray Data passes to each map_batches invocation."
        ),
    )
    num_cpus: float = Field(
        default=1.0,
        ge=0,
        description=(
            "CPUs reserved per execution unit (per Ray Data task in "
            "compute='tasks' mode, per actor in compute='actors' mode)."
        ),
    )
    num_gpus: float = Field(
        default=0.0,
        ge=0,
        description=(
            "GPUs reserved per execution unit (per Ray Data task in "
            "compute='tasks' mode, per actor in compute='actors' mode)."
        ),
    )
    compute: Literal["tasks", "actors"] = Field(
        default="tasks",
        description=(
            "Execution mode.  'tasks' (default) runs each batch as a "
            "stateless Ray Data task with a worker-local backend cache, "
            "so load() fires once per worker process.  'actors' uses an "
            "actor pool sized by num_actors; load() fires once per actor "
            "at __init__ and persists for the actor's lifetime, "
            "eliminating cold-start cost on subsequent batches.  Use "
            "actors for backends with expensive load() (FLUX, GraphCast, "
            "SDXL); use tasks for cheap loads or small jobs."
        ),
    )
    num_actors: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Actor pool size when compute='actors'.  Required in actor "
            "mode, ignored in task mode.  Each actor reserves num_cpus + "
            "num_gpus for its lifetime; sizing must satisfy "
            "num_actors * num_gpus <= cluster GPUs."
        ),
    )

    @model_validator(mode="after")
    def _validate_compute_config(self) -> BatchSpec:
        if self.compute == "actors" and self.num_actors is None:
            raise ValueError(
                "num_actors is required when compute='actors'.  Set "
                "num_actors=N to size the pool (e.g. num_actors=2 with "
                "num_gpus=1 reserves 2 GPUs)."
            )
        return self
