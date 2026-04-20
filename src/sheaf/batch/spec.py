"""BatchSpec — declares what Sheaf should run as an offline batch job.

Mirrors ``ModelSpec`` for the backend-selection fields (``name``,
``model_type``, ``backend``, ``backend_cls``, ``backend_kwargs``) and adds
``source`` + ``sink`` + execution config for offline pipelines.

v0.6 ships only JSONL I/O.  S3 / Parquet / Delta sources and sinks are
tracked in follow-up issues and will slot in as additional
``BatchSource``/``BatchSink`` subclasses without changing the runner API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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
        description="CPUs reserved per Ray Data task.",
    )
    num_gpus: float = Field(
        default=0.0,
        ge=0,
        description="GPUs reserved per Ray Data task.",
    )
