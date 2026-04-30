"""Async-job worker quickstart — Redis Streams queue + hash result store.

Decouples request submission from inference for jobs where HTTP
request/response is the wrong shape (FLUX 50-step, GraphCast multi-day
rollouts, large-batch SDXL).

Architecture::

    client.enqueue(req) ──► Redis Stream ──► SheafWorker ──► result store
                                                  │
                                                  └─► optional webhook POST

Run a worker process::

    pip install 'sheaf-serve[time-series,worker]'
    redis-server &  # or any Redis ≥5
    python examples/quickstart_worker.py worker

In a separate shell, enqueue a job and wait for the result::

    python examples/quickstart_worker.py client

This example uses ``amazon/chronos-bolt-tiny`` so it runs on CPU.  For
your real workloads, swap in any registered backend ("flux", "sdxl",
"graphcast", etc.) and let the worker shoulder the long inferences.
"""

from __future__ import annotations

import sys
import time

from sheaf.api.base import ModelType
from sheaf.worker import (
    JobQueueClient,
    RedisHashResultStore,
    RedisStreamsQueue,
    SheafWorker,
    WorkerSpec,
)

REDIS_URL = "redis://localhost:6379/0"
STREAM = "sheaf:demo:chronos"
GROUP = "demo-workers"
RESULT_PREFIX = "sheaf:demo:chronos:result"


def _make_queue(consumer: str) -> RedisStreamsQueue:
    return RedisStreamsQueue(
        stream=STREAM, group=GROUP, consumer=consumer, url=REDIS_URL
    )


def _make_results() -> RedisHashResultStore:
    return RedisHashResultStore(prefix=RESULT_PREFIX, url=REDIS_URL, ttl_seconds=3600)


def run_worker() -> None:
    spec = WorkerSpec(
        name="chronos-worker",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        backend_kwargs={
            "model_id": "amazon/chronos-bolt-tiny",
            "device_map": "cpu",
            "torch_dtype": "float32",
        },
        queue=_make_queue(consumer="worker-1"),
        results=_make_results(),
        max_retries=2,
    )
    print(f"Starting worker {spec.name!r} (Ctrl-C to stop)…")
    SheafWorker(spec).start()


def run_client() -> None:
    queue = _make_queue(consumer="client-only")
    results = _make_results()
    client = JobQueueClient(queue, results)

    request = {
        "model_type": "time_series",
        "model_name": "demand-zone-01",
        "history": [312, 298, 275, 260, 255, 263, 285, 320, 368, 402, 421, 435],
        "horizon": 6,
        "frequency": "1h",
        "output_mode": "mean",
    }

    t0 = time.time()
    job_id = client.enqueue(request)
    print(f"enqueued job_id={job_id}; waiting for result…")
    result = client.wait_for_result(job_id, timeout_s=120.0, poll_interval_s=0.5)
    dt = time.time() - t0

    print(f"\nstatus={result.status}  ({dt:.2f}s)")
    if result.response is not None:
        print(f"forecast mean={[round(x, 1) for x in result.response['mean']]}")
    if result.error is not None:
        print(f"error: {result.error}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"worker", "client"}:
        print("usage: python examples/quickstart_worker.py [worker|client]")
        sys.exit(2)
    if sys.argv[1] == "worker":
        run_worker()
    else:
        run_client()
