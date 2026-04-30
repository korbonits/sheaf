"""Async job-queue worker — long-running inference decoupled from HTTP.

For jobs where request/response is the wrong shape (FLUX 50-step,
GraphCast multi-day rollouts, batch SDXL): clients ``enqueue`` a
typed request and either poll the result store or wait for a webhook.

Public surface::

    from sheaf.worker import (
        SheafWorker,
        WorkerSpec,
        JobQueueClient,
        JobQueue,
        ResultStore,
        RedisStreamsQueue,
        RedisHashResultStore,
        Job,
        JobResult,
    )

Install with: pip install 'sheaf-serve[worker]'
"""

from sheaf.worker.queue import (
    Job,
    JobQueue,
    JobQueueClient,
    JobResult,
    RedisHashResultStore,
    RedisStreamsQueue,
    ResultStore,
)
from sheaf.worker.runner import SheafWorker
from sheaf.worker.spec import WorkerSpec

__all__ = [
    "Job",
    "JobQueue",
    "JobQueueClient",
    "JobResult",
    "RedisHashResultStore",
    "RedisStreamsQueue",
    "ResultStore",
    "SheafWorker",
    "WorkerSpec",
]
