"""WorkerSpec — declares an async-job worker process.

Mirrors ``ModelSpec``/``BatchSpec`` for backend selection (``name``,
``model_type``, ``backend``, ``backend_cls``, ``backend_kwargs``) and adds
queue/result-store wiring + retry/webhook policy.

v0.7 ships only Redis Streams + Redis Hash adapters.  SQS / Kafka /
Postgres slot in as additional ``JobQueue`` / ``ResultStore`` subclasses
without changing the worker loop.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sheaf.api.base import ModelType


class WorkerSpec(BaseModel):
    """Declares an async worker process consuming jobs from a queue.

    Example::

        from sheaf.worker import WorkerSpec, RedisStreamsQueue, RedisHashResultStore

        spec = WorkerSpec(
            name="flux-worker",
            model_type=ModelType.DIFFUSION,
            backend="flux",
            backend_kwargs={"model_id": "black-forest-labs/FLUX.1-schnell"},
            queue=RedisStreamsQueue(
                stream="sheaf:flux", group="workers", consumer="w1"
            ),
            results=RedisHashResultStore(prefix="sheaf:flux:result"),
            max_retries=3,
        )
        SheafWorker(spec).start()

    ``backend`` / ``backend_cls`` / ``backend_kwargs`` follow the same
    semantics as on ``ModelSpec``: ``backend_cls`` takes precedence over
    the registry lookup when set.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    model_type: ModelType
    backend: str
    # type[ModelBackend] | None — Any avoids Pydantic schema generation issues
    backend_cls: Any = None
    backend_kwargs: dict = Field(default_factory=dict)

    # Queue + result store. Use Any so the spec module doesn't import the
    # queue module (and through it, redis) at import time. Worker module
    # imports both.
    queue: Any
    results: Any

    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Max number of times a job is re-delivered after a worker "
            "exception before it is moved to the dead-letter stream.  0 "
            "means a single attempt with no retries."
        ),
    )
    poll_block_ms: int = Field(
        default=5000,
        gt=0,
        description=(
            "Max time the worker blocks on the queue waiting for a new "
            "job before looping (lets SIGINT take effect).  Lower values "
            "mean faster shutdown but more idle CPU."
        ),
    )
    webhook_timeout_s: float = Field(
        default=10.0,
        gt=0,
        description="Per-webhook HTTP timeout.  Failures are logged, not retried.",
    )
