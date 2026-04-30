"""SheafWorker — async-job consumer loop.

Single-threaded loop:

    1. ``queue.dequeue(block_ms)``  → ``Job`` or ``None``
    2. ``backend.predict(request)`` (validated against ``AnyRequest`` first)
    3. ``results.put(JobResult(status="completed", ...))``
    4. POST webhook if ``webhook_url`` is set (failures logged, not raised)
    5. ``queue.ack(job)``  — only after step 3 succeeds, so a crash between
       infer and ack causes redelivery (at-least-once on driver crash).

On exception in step 2 the worker increments a local delivery counter
keyed by ``job_id``; once it exceeds ``spec.max_retries`` the job is
moved to the dead-letter queue and a ``status="failed"`` ``JobResult``
is written so :meth:`JobQueueClient.wait_for_result` doesn't block
forever.

Reuses ``sheaf.metrics.record_predict`` (same labels as the Ray Serve
hot-path), and emits OTel spans ``sheaf.worker.process`` /
``sheaf.worker.backend.infer`` / ``sheaf.worker.webhook`` via
``trace_span``.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

from pydantic import TypeAdapter

from sheaf.api.union import AnyRequest
from sheaf.backends.base import ModelBackend
from sheaf.metrics import record_predict
from sheaf.tracing import record_exception, trace_span
from sheaf.worker.queue import Job, JobResult
from sheaf.worker.spec import WorkerSpec

_logger = logging.getLogger(__name__)
_REQUEST_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyRequest)


def _build_backend(spec: WorkerSpec) -> ModelBackend:
    # Same registry-refresh dance as BatchRunner — workers (here, just
    # the worker process itself) shouldn't trust module-level imports.
    from sheaf.backends._register import (
        register_builtin_backends,
        register_extra_backends,
    )

    register_builtin_backends()
    register_extra_backends()

    from sheaf.registry import _BACKEND_REGISTRY as _registry

    backend_cls = spec.backend_cls or _registry.get(spec.backend)
    if backend_cls is None:
        raise ValueError(
            f"Unknown backend '{spec.backend}'. Registered backends: {list(_registry)}"
        )
    backend: ModelBackend = backend_cls(**spec.backend_kwargs)
    backend.load()
    return backend


class SheafWorker:
    """Async-job consumer for any ``ModelBackend``.

    Example::

        from sheaf.api.base import ModelType
        from sheaf.worker import (
            SheafWorker, WorkerSpec, RedisStreamsQueue, RedisHashResultStore,
        )

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
        SheafWorker(spec).start()  # blocks until SIGINT
    """

    def __init__(self, spec: WorkerSpec, backend: ModelBackend | None = None) -> None:
        self._spec = spec
        self._backend = backend or _build_backend(spec)
        # Local delivery counters for retry decisions.  Redis Streams
        # tracks pending count too, but we keep our own so the abstraction
        # works for queues without that capability.
        self._delivery_counts: dict[str, int] = {}
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the consume loop until SIGINT/SIGTERM.

        Blocks the calling thread.  After receiving a stop signal, the
        worker finishes its current job (if any) before returning.
        """
        self._install_signal_handlers()
        _logger.info(
            "SheafWorker %s starting (backend=%s, model_type=%s)",
            self._spec.name,
            self._spec.backend,
            self._spec.model_type,
            extra={
                "deployment": self._spec.name,
                "model_type": str(self._spec.model_type),
                "backend": self._spec.backend,
            },
        )
        while not self._stop_requested:
            self.run_one()
        _logger.info(
            "SheafWorker %s stopped",
            self._spec.name,
            extra={"deployment": self._spec.name},
        )

    def stop(self) -> None:
        """Request graceful shutdown.  The current job, if any, finishes first."""
        self._stop_requested = True

    def run_one(self) -> Job | None:
        """Process at most one job.  Returns the ``Job`` processed, or ``None`` if idle.

        Exposed publicly so tests can step the loop one iteration at a
        time without spinning a real signal handler.
        """
        with trace_span("sheaf.worker.dequeue", deployment=self._spec.name):
            job = self._spec.queue.dequeue(block_ms=self._spec.poll_block_ms)
        if job is None:
            return None
        # Track delivery count locally — increment on each receipt of this job_id.
        self._delivery_counts[job.job_id] = self._delivery_counts.get(job.job_id, 0) + 1
        job.delivery_count = self._delivery_counts[job.job_id]
        self._process(job)
        return job

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process(self, job: Job) -> None:
        t0 = time.time()
        spec = self._spec

        with trace_span(
            "sheaf.worker.process",
            deployment=spec.name,
            job_id=job.job_id,
            delivery_count=job.delivery_count,
        ) as span:
            try:
                request = _REQUEST_ADAPTER.validate_python(job.request)
                if request.model_type != spec.model_type:
                    raise ValueError(
                        f"job {job.job_id} model_type={request.model_type!r} "
                        f"does not match WorkerSpec.model_type={spec.model_type!r}"
                    )

                with trace_span("sheaf.worker.backend.infer", deployment=spec.name):
                    response = self._backend.predict(request)

                self._spec.results.put(
                    JobResult(
                        job_id=job.job_id,
                        status="completed",
                        response=response.model_dump(mode="json"),
                        completed_at=time.time(),
                    )
                )
                self._maybe_webhook(
                    job,
                    status="completed",
                    payload=response.model_dump(mode="json"),
                    error=None,
                )
                self._spec.queue.ack(job)
                self._delivery_counts.pop(job.job_id, None)

                latency = time.time() - t0
                record_predict(spec.name, str(spec.model_type), "ok", latency)
                _logger.info(
                    "job ok",
                    extra={
                        "deployment": spec.name,
                        "model_type": str(spec.model_type),
                        "job_id": job.job_id,
                        "latency_ms": round(latency * 1000, 3),
                        "status": "ok",
                    },
                )

            except Exception as exc:
                record_exception(span, exc)
                latency = time.time() - t0
                record_predict(spec.name, str(spec.model_type), "error", latency)
                self._handle_failure(job, exc, latency)

    def _handle_failure(self, job: Job, exc: Exception, latency: float) -> None:
        spec = self._spec
        retries_remaining = spec.max_retries - job.delivery_count
        # delivery_count starts at 1; max_retries=3 means try the original
        # delivery + 3 redeliveries == 4 total attempts.
        if retries_remaining > 0:
            _logger.warning(
                "job error — will retry",
                extra={
                    "deployment": spec.name,
                    "job_id": job.job_id,
                    "delivery_count": job.delivery_count,
                    "retries_remaining": retries_remaining,
                    "error": str(exc),
                    "latency_ms": round(latency * 1000, 3),
                    "status": "retry",
                },
            )
            spec.queue.nack(job)
            return

        # Out of retries — dead-letter and write a failed result so callers
        # waiting on the result store don't hang forever.
        reason = f"{type(exc).__name__}: {exc}"
        spec.queue.dead_letter(job, reason=reason)
        spec.results.put(
            JobResult(
                job_id=job.job_id,
                status="failed",
                error=reason,
                completed_at=time.time(),
            )
        )
        self._maybe_webhook(job, status="failed", payload=None, error=reason)
        self._delivery_counts.pop(job.job_id, None)
        _logger.exception(
            "job failed — dead-lettered",
            extra={
                "deployment": spec.name,
                "job_id": job.job_id,
                "delivery_count": job.delivery_count,
                "error": reason,
                "latency_ms": round(latency * 1000, 3),
                "status": "failed",
            },
        )

    def _maybe_webhook(
        self,
        job: Job,
        status: str,
        payload: dict | None,
        error: str | None,
    ) -> None:
        if not job.webhook_url:
            return
        body = {
            "job_id": job.job_id,
            "status": status,
            "response": payload,
            "error": error,
        }
        with trace_span(
            "sheaf.worker.webhook",
            deployment=self._spec.name,
            job_id=job.job_id,
            url=job.webhook_url,
        ):
            try:
                import httpx  # ty: ignore[unresolved-import]

                httpx.post(
                    job.webhook_url,
                    json=body,
                    timeout=self._spec.webhook_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                # Webhooks are best-effort — log and move on.  The result is
                # already in the result store; downstream can poll if it cares.
                _logger.warning(
                    "webhook failed",
                    extra={
                        "deployment": self._spec.name,
                        "job_id": job.job_id,
                        "url": job.webhook_url,
                        "error": str(exc),
                    },
                )

    def _install_signal_handlers(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            _logger.info(
                "SheafWorker %s received signal %d; finishing current job",
                self._spec.name,
                signum,
                extra={"deployment": self._spec.name},
            )
            self._stop_requested = True

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            # signal.signal only works in main thread; tests/threads skip.
            pass
