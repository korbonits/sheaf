"""Queue + result-store abstractions and the Redis Streams v1 adapter.

Two ABCs:

* ``JobQueue``: ``enqueue`` / ``dequeue`` / ``ack`` / ``nack`` / ``dead_letter``.
* ``ResultStore``: ``put`` / ``get``.

A ``Job`` is a typed envelope around an ``AnyRequest`` plus delivery
metadata (``job_id``, ``msg_id``, ``delivery_count``, optional
``webhook_url``).  ``JobResult`` is the success/failure record persisted
to the result store.

Redis Streams + Hash is the v1 backend.  XADD enqueues, XREADGROUP
claims under a consumer group (so multiple workers scale horizontally),
XACK confirms processing, and a separate ``dead_letter`` stream
captures jobs that have exceeded ``max_retries``.

``JobQueueClient`` is the small client-side helper for application code:
``enqueue(request, webhook_url=None) -> job_id`` and
``wait_for_result(job_id, timeout)`` (poll the result store).
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, cast

from pydantic import BaseModel, Field, TypeAdapter

from sheaf.api.union import AnyRequest

_REQUEST_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyRequest)


class Job(BaseModel):
    """A dequeued job — request + delivery metadata."""

    model_config = {"arbitrary_types_allowed": True}

    job_id: str
    msg_id: str  # opaque queue-side id used for ack/nack/dead_letter
    request: dict  # raw AnyRequest dict, validated by the worker before infer
    webhook_url: str | None = None
    submitted_at: float
    delivery_count: int = Field(
        default=1,
        description=(
            "Number of times this job has been delivered.  Increments on "
            "redelivery after a worker crash or nack; the worker compares "
            "against ``max_retries`` to decide dead-letter."
        ),
    )


class JobResult(BaseModel):
    """Result record written to the result store."""

    job_id: str
    status: str  # "completed" | "failed"
    response: dict | None = None
    error: str | None = None
    completed_at: float


class JobQueue(ABC):
    """Abstract job queue.  Subclass to add new backends (SQS, Kafka, ...)."""

    @abstractmethod
    def enqueue(
        self,
        request: dict,
        webhook_url: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Add a job to the queue.  Returns the ``job_id``."""
        ...

    @abstractmethod
    def dequeue(self, block_ms: int) -> Job | None:
        """Block up to ``block_ms`` for the next job.  Returns ``None`` on timeout."""
        ...

    @abstractmethod
    def ack(self, job: Job) -> None:
        """Confirm successful processing — removes the job from in-flight tracking."""
        ...

    @abstractmethod
    def nack(self, job: Job) -> None:
        """Mark the job as failed — leaves it for retry by the consumer group."""
        ...

    @abstractmethod
    def dead_letter(self, job: Job, reason: str) -> None:
        """Move the job to the dead-letter queue with a failure reason."""
        ...

    def queue_depth(self) -> int | None:
        """Approximate pending+inflight count, or ``None`` if unsupported."""
        return None


class ResultStore(ABC):
    """Abstract result store.  Subclass to add new backends (S3, Postgres, ...)."""

    @abstractmethod
    def put(self, result: JobResult) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> JobResult | None: ...


# ---------------------------------------------------------------------------
# Redis Streams + Hash — v1 backend
# ---------------------------------------------------------------------------


class RedisStreamsQueue(JobQueue):
    """Redis Streams + consumer groups job queue.

    Multiple workers using the same ``stream`` + ``group`` form a
    horizontal-scaling consumer pool: each job is delivered to exactly
    one consumer.  Jobs that exceed ``max_retries`` (decided by the
    worker, not the queue) are XADDed to a separate dead-letter stream
    via :meth:`dead_letter`.

    Notes:
        * The stream and the consumer group are created lazily on first
          use (idempotent ``XGROUP CREATE ... MKSTREAM``).
        * ``XACK`` is the only way a job leaves "pending" state for this
          consumer; the worker calls :meth:`ack` only after the result
          is persisted to the result store, so a crash between infer and
          ack causes redelivery (at-least-once).
        * ``redis-py`` is imported lazily — the ``[worker]`` extra adds
          ``redis>=5.0`` but ``sheaf.worker.queue`` should be importable
          for type-only usage without it.
    """

    def __init__(
        self,
        stream: str,
        group: str,
        consumer: str,
        url: str = "redis://localhost:6379/0",
        dead_letter_stream: str | None = None,
        client: Any = None,
    ) -> None:
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._dead_letter_stream = dead_letter_stream or f"{stream}:dead"

        if client is not None:
            # Test injection.
            self._redis = client
        else:
            try:
                import redis  # ty: ignore[unresolved-import]
            except ImportError as e:
                raise ImportError(
                    "RedisStreamsQueue requires redis. "
                    "Install with: pip install 'sheaf-serve[worker]'"
                ) from e
            self._redis = redis.Redis.from_url(url, decode_responses=True)

        self._ensure_group()

    def _ensure_group(self) -> None:
        try:
            self._redis.xgroup_create(
                name=self._stream, groupname=self._group, id="0", mkstream=True
            )
        except Exception as e:  # noqa: BLE001
            # BUSYGROUP — group already exists; idempotent.
            if "BUSYGROUP" not in str(e):
                raise

    def enqueue(
        self,
        request: dict,
        webhook_url: str | None = None,
        job_id: str | None = None,
    ) -> str:
        jid = job_id or str(uuid.uuid4())
        payload = {
            "job_id": jid,
            "request": json.dumps(request),
            "webhook_url": webhook_url or "",
            "submitted_at": str(time.time()),
        }
        # redis-py types `fields` as dict[BytesLike, BytesLike]; pass through
        # cast(Any) so ty doesn't reject our str/str payload (decode_responses
        # makes the round-trip work in either direction).
        cast(Any, self._redis).xadd(self._stream, payload)
        return jid

    def dequeue(self, block_ms: int) -> Job | None:
        # ">" — only new (never-delivered-to-this-consumer) entries.
        resp = cast(
            Any,
            self._redis.xreadgroup(
                groupname=self._group,
                consumername=self._consumer,
                streams={self._stream: ">"},
                count=1,
                block=block_ms,
            ),
        )
        if not resp:
            return None
        # resp = [(stream_name, [(msg_id, {field: value, ...})])]
        _stream, entries = resp[0]
        msg_id, fields = entries[0]
        return Job(
            job_id=fields["job_id"],
            msg_id=msg_id,
            request=json.loads(fields["request"]),
            webhook_url=fields.get("webhook_url") or None,
            submitted_at=float(fields["submitted_at"]),
            # Pending-entry-list lookups would give the true delivery count;
            # for v1 we trust the worker's local counter.  Default 1 here.
            delivery_count=1,
        )

    def ack(self, job: Job) -> None:
        self._redis.xack(self._stream, self._group, job.msg_id)

    def nack(self, job: Job) -> None:
        # Redis Streams keeps the entry in PEL (pending-entry-list) until
        # XACKed.  Leaving it there is the redelivery mechanism.  No-op.
        return

    def dead_letter(self, job: Job, reason: str) -> None:
        self._redis.xadd(
            self._dead_letter_stream,
            {
                "job_id": job.job_id,
                "request": json.dumps(job.request),
                "webhook_url": job.webhook_url or "",
                "submitted_at": str(job.submitted_at),
                "delivery_count": str(job.delivery_count),
                "reason": reason,
            },
        )
        # Ack the original so it stops redelivering.
        self.ack(job)

    def queue_depth(self) -> int | None:
        # redis-py exposes async/sync via overload, so the return types come
        # back as `Awaitable[T] | T` — cast to the sync side we know is true
        # here (decode_responses=True implies the synchronous client).
        try:
            length = cast(int, self._redis.xlen(self._stream))
            pending = cast(Any, self._redis.xpending(self._stream, self._group))
            pending_count = (
                pending.get("pending", 0) if isinstance(pending, dict) else 0
            )
            return int(length) + int(pending_count)
        except Exception:  # noqa: BLE001
            return None


class RedisHashResultStore(ResultStore):
    """Redis hash-per-job result store.

    Each result is stored under the key ``f"{prefix}:{job_id}"`` as a
    Redis hash with fields ``status``, ``response``, ``error``,
    ``completed_at``.  Optional TTL (``ttl_seconds``) lets results
    auto-expire so the store doesn't grow unbounded.
    """

    def __init__(
        self,
        prefix: str = "sheaf:result",
        url: str = "redis://localhost:6379/0",
        ttl_seconds: int | None = 86400,
        client: Any = None,
    ) -> None:
        self._prefix = prefix
        self._ttl = ttl_seconds
        if client is not None:
            self._redis = client
        else:
            try:
                import redis  # ty: ignore[unresolved-import]
            except ImportError as e:
                raise ImportError(
                    "RedisHashResultStore requires redis. "
                    "Install with: pip install 'sheaf-serve[worker]'"
                ) from e
            self._redis = redis.Redis.from_url(url, decode_responses=True)

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}:{job_id}"

    def put(self, result: JobResult) -> None:
        key = self._key(result.job_id)
        mapping = {
            "status": result.status,
            "completed_at": str(result.completed_at),
            "response": json.dumps(result.response)
            if result.response is not None
            else "",
            "error": result.error or "",
        }
        self._redis.hset(key, mapping=mapping)
        if self._ttl is not None:
            self._redis.expire(key, self._ttl)

    def get(self, job_id: str) -> JobResult | None:
        data = cast(dict, self._redis.hgetall(self._key(job_id)))
        if not data:
            return None
        return JobResult(
            job_id=job_id,
            status=data["status"],
            response=json.loads(data["response"]) if data.get("response") else None,
            error=data.get("error") or None,
            completed_at=float(data["completed_at"]),
        )


# ---------------------------------------------------------------------------
# Client-side helper
# ---------------------------------------------------------------------------


class JobQueueClient:
    """Application-side helper: enqueue jobs, optionally wait for results.

    Wraps a ``JobQueue`` (for enqueue) and a ``ResultStore`` (for poll).
    Validates the request against ``AnyRequest`` before enqueue so
    callers see schema errors immediately, not in a worker log later.
    """

    def __init__(self, queue: JobQueue, results: ResultStore) -> None:
        self._queue = queue
        self._results = results

    def enqueue(
        self,
        request: dict | BaseModel,
        webhook_url: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Validate + submit a request.  Returns the assigned ``job_id``."""
        if isinstance(request, BaseModel):
            req_dict = request.model_dump(mode="json")
        else:
            req_dict = request
        # Validate up-front — schema errors should never reach a worker.
        _REQUEST_ADAPTER.validate_python(req_dict)
        return self._queue.enqueue(req_dict, webhook_url=webhook_url, job_id=job_id)

    def wait_for_result(
        self, job_id: str, timeout_s: float = 60.0, poll_interval_s: float = 0.25
    ) -> JobResult:
        """Poll the result store until the job completes or ``timeout_s`` elapses.

        Raises ``TimeoutError`` if the deadline passes with no result.
        """
        deadline = time.time() + timeout_s
        while True:
            result = self._results.get(job_id)
            if result is not None:
                return result
            if time.time() >= deadline:
                raise TimeoutError(f"job {job_id} did not complete in {timeout_s}s")
            time.sleep(poll_interval_s)
