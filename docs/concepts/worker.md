# Async-job worker

For jobs that take longer than a synchronous HTTP request can wait
for — minutes-long batch transcripts, multi-step generative pipelines,
anything that needs retry-on-failure or fan-out — sheaf-serve ships
`SheafWorker`: a background process that consumes from a queue and
writes results to a key/value store, with optional webhook callbacks.

The v1 backend is **Redis Streams + consumer groups**. The
`JobQueue` / `ResultStore` ABCs let SQS / Kafka / Postgres slot in
later without touching the worker logic.

```bash
pip install "sheaf-serve[worker,time-series]"
```

## Submitting jobs

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.api.time_series import TimeSeriesRequest, Frequency
from sheaf.worker import (
    JobQueueClient,
    RedisHashResultStore,
    RedisStreamsQueue,
)

queue = RedisStreamsQueue(url="redis://localhost:6379", stream="jobs:forecaster")
results = RedisHashResultStore(url="redis://localhost:6379", key_prefix="job:")

client = JobQueueClient(queue=queue, results=results)

job_id = client.enqueue(TimeSeriesRequest(
    model_name="forecaster",
    history=[100, 110, 120, ...],
    horizon=12,
    frequency=Frequency.HOURLY,
))

# Block until result lands or timeout fires
result = client.wait_for_result(job_id, timeout_s=300)
print(result.status, result.response)
```

## Running the worker

In a separate process (typically a long-running container):

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.worker import RedisStreamsQueue, RedisHashResultStore, SheafWorker, WorkerSpec

spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={"model_id": "amazon/chronos-bolt-small"},
)

worker = SheafWorker(WorkerSpec(
    model_spec=spec,
    queue=RedisStreamsQueue(url="redis://localhost:6379", stream="jobs:forecaster"),
    results=RedisHashResultStore(url="redis://localhost:6379", key_prefix="job:"),
    max_retries=3,
))
worker.run()  # blocks; SIGINT / SIGTERM trigger graceful shutdown
```

## Delivery semantics

**At-least-once.** The consume loop is:

```
dequeue → predict → results.put → webhook → ack
```

XACK is the only signal Redis Streams uses to remove a job from the
consumer group's pending-entry list, so a worker crash between
`predict` and `ack` causes the job to be redelivered to another
consumer. This guarantees at-least-once; backend code that produces
side effects must be idempotent, or the result hash should be the
source of truth.

## Retries + dead-letter

Each `dequeue` of the same job ID increments a process-local delivery
counter. When it exceeds `max_retries`, the worker XADDs the job to
the dead-letter stream **and** writes a `JobResult(status="failed",
error=...)` to the result store. Without the result-store write,
`wait_for_result(job_id, timeout_s)` would block until timeout for
every poison-pill job — the dead-letter stream is for ops/triage; the
result store is for waiting clients.

## Webhook callbacks

`JobQueueClient.enqueue(req, webhook_url=...)` records a callback on
the job. The worker POSTs the result to `webhook_url` between
`results.put` and `queue.ack`. Webhook failures are caught and logged,
not raised — the result is already in the store; downstream can poll.

## Metrics

`SheafWorker` reuses `sheaf.metrics.record_predict(spec.name,
model_type, status, latency)`. Same Prometheus series shape as the
HTTP path (`sheaf_requests_total{deployment, model_type, status}`);
workers are distinguished only by the `deployment` label value. No
dashboard drift.

## Reference

Full schema in the [Worker API reference](../api-reference/worker.md).
