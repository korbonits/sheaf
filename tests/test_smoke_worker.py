"""End-to-end smoke for SheafWorker against a real Redis client.

Exercises the actual ``RedisStreamsQueue`` + ``RedisHashResultStore``
code paths (XADD / XREADGROUP / XACK / HSET / HGETALL / dead-letter
XADD), unlike ``test_worker.py`` which uses in-memory ``FakeQueue``
fakes.

Two run modes:

* **Default**: uses ``fakeredis.FakeStrictRedis`` injected via the
  ``client=`` parameter — works in CI without any service container,
  while still exercising every Redis command the adapter issues.

* **Real Redis**: set ``SHEAF_INTEGRATION_REDIS_URL=redis://...`` to
  connect a live Redis (e.g. ``redis://localhost:6379/0``) instead.
  The whole test module re-runs against that backend.

Either way, the worker, queue, result store, and client are the
production classes — only the Redis client itself is swapped.
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

fakeredis = pytest.importorskip("fakeredis")

from sheaf.api.base import ModelType  # noqa: E402
from sheaf.worker import (  # noqa: E402
    JobQueueClient,
    RedisHashResultStore,
    RedisStreamsQueue,
    SheafWorker,
    WorkerSpec,
)

# Register smoke backends in this process.
os.environ.setdefault("SHEAF_EXTRA_BACKENDS", "tests.stubs")
import tests.stubs  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Redis-client fixture — fakeredis by default, real Redis via env var
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_client():
    real_url = os.environ.get("SHEAF_INTEGRATION_REDIS_URL")
    if real_url:
        import redis

        client = redis.Redis.from_url(real_url, decode_responses=True)
        # Sanity-ping so we fail fast if the URL is unreachable.
        client.ping()
        yield client
        # No flush — leave a real Redis untouched, just rely on unique stream
        # names below.
    else:
        client = fakeredis.FakeStrictRedis(decode_responses=True)
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


def _ts_request() -> dict:
    return {
        "model_type": "time_series",
        "model_name": "smoke",
        "history": [1.0, 2.0, 3.0, 4.0],
        "horizon": 3,
        "frequency": "1h",
    }


def _make_queue(redis_client, *, suffix: str) -> RedisStreamsQueue:
    return RedisStreamsQueue(
        stream=f"sheaf:test:{suffix}",
        group="g",
        consumer="c1",
        client=redis_client,
    )


def _make_results(redis_client, *, suffix: str) -> RedisHashResultStore:
    return RedisHashResultStore(
        prefix=f"sheaf:test:result:{suffix}",
        ttl_seconds=None,  # don't EXPIRE during a unit test
        client=redis_client,
    )


def _make_spec(queue, results, **overrides) -> WorkerSpec:
    base = {
        "name": "smoke-worker",
        "model_type": ModelType.TIME_SERIES,
        "backend": "_smoke_ts",
        "queue": queue,
        "results": results,
        "max_retries": 1,
        "poll_block_ms": 50,
    }
    base.update(overrides)
    return WorkerSpec(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_enqueue_consume_result_round_trip(redis_client):
    suffix = _unique_suffix()
    queue = _make_queue(redis_client, suffix=suffix)
    results = _make_results(redis_client, suffix=suffix)
    client = JobQueueClient(queue, results)

    # Real XADD goes here.
    job_id = client.enqueue(_ts_request())

    # Stream length reflects the enqueue.
    assert redis_client.xlen(f"sheaf:test:{suffix}") == 1

    # One worker step pulls the job, runs the smoke backend, writes
    # the result hash, and XACKs.
    worker = SheafWorker(_make_spec(queue, results))
    job = worker.run_one()
    assert job is not None and job.job_id == job_id

    # Result store sees the completed job.
    result = client.wait_for_result(job_id, timeout_s=1.0, poll_interval_s=0.01)
    assert result.status == "completed"
    assert result.response is not None
    assert result.response["mean"] == [0.42, 0.42, 0.42]

    # XACK happened — pending count is zero.
    pending = redis_client.xpending(f"sheaf:test:{suffix}", "g")
    pending_count = pending.get("pending", 0) if isinstance(pending, dict) else 0
    assert pending_count == 0


def test_dead_letter_after_retries_writes_failed_result_and_dl_stream(redis_client):
    suffix = _unique_suffix()
    queue = _make_queue(redis_client, suffix=suffix)
    results = _make_results(redis_client, suffix=suffix)
    client = JobQueueClient(queue, results)

    spec = _make_spec(
        queue,
        results,
        backend="_smoke_error",  # raises on every predict
        max_retries=1,
    )

    job_id = client.enqueue(_ts_request())

    worker = SheafWorker(spec)
    # delivery 1 → retries_remaining=0 → dead-letter.
    worker.run_one()

    # Dead-letter stream got the entry.
    dl_stream = f"sheaf:test:{suffix}:dead"
    assert redis_client.xlen(dl_stream) == 1
    dl_entries = redis_client.xrange(dl_stream)
    assert dl_entries[0][1]["job_id"] == job_id
    assert "backend exploded" in dl_entries[0][1]["reason"]

    # Result store has a failed JobResult so wait_for_result doesn't hang.
    result = client.wait_for_result(job_id, timeout_s=1.0, poll_interval_s=0.01)
    assert result.status == "failed"
    assert "backend exploded" in (result.error or "")


def test_two_workers_share_via_consumer_group(redis_client):
    """Two SheafWorker instances on the same stream/group split the work."""
    suffix = _unique_suffix()

    # Two queues bound to the *same* stream + group, different consumer names.
    q_a = RedisStreamsQueue(
        stream=f"sheaf:test:{suffix}",
        group="g",
        consumer="worker-a",
        client=redis_client,
    )
    q_b = RedisStreamsQueue(
        stream=f"sheaf:test:{suffix}",
        group="g",
        consumer="worker-b",
        client=redis_client,
    )
    results = _make_results(redis_client, suffix=suffix)

    # Submit four jobs through one of the queues.
    client = JobQueueClient(q_a, results)
    job_ids = [client.enqueue(_ts_request()) for _ in range(4)]

    # Each worker drains until empty.
    w_a = SheafWorker(_make_spec(q_a, results, name="w-a"))
    w_b = SheafWorker(_make_spec(q_b, results, name="w-b"))

    drained_a, drained_b = [], []
    for _ in range(8):  # at most 8 steps; expect 4 to land
        job = w_a.run_one()
        if job is not None:
            drained_a.append(job.job_id)
        job = w_b.run_one()
        if job is not None:
            drained_b.append(job.job_id)
        if len(drained_a) + len(drained_b) >= 4:
            break

    drained = drained_a + drained_b
    assert sorted(drained) == sorted(job_ids), (
        f"jobs drained={drained}, expected={job_ids}"
    )
    # And each worker actually got at least one — i.e. consumer-group
    # dispatch really split work.  (FakeStrictRedis preserves XREADGROUP
    # round-robin behavior across consumers in the same group.)
    assert drained_a, "worker-a got no jobs — consumer group did not split"
    assert drained_b, "worker-b got no jobs — consumer group did not split"


def test_signal_based_graceful_shutdown_in_thread(redis_client):
    """SheafWorker.start() in a thread; .stop() drains and exits cleanly."""
    suffix = _unique_suffix()
    queue = _make_queue(redis_client, suffix=suffix)
    results = _make_results(redis_client, suffix=suffix)
    client = JobQueueClient(queue, results)

    spec = _make_spec(queue, results, poll_block_ms=20)
    worker = SheafWorker(spec)

    # Submit one job, then start the worker in a thread.
    job_id = client.enqueue(_ts_request())

    t = threading.Thread(target=worker.start, daemon=True)
    t.start()

    # Wait for the result to appear (worker processed at least one job).
    result = client.wait_for_result(job_id, timeout_s=2.0, poll_interval_s=0.01)
    assert result.status == "completed"

    # Stop the worker; it should exit within a couple poll cycles.
    worker.stop()
    t.join(timeout=2.0)
    assert not t.is_alive(), "worker did not exit after stop()"
