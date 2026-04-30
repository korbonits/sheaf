"""Tests for sheaf.worker — spec, FakeQueue/FakeResultStore, SheafWorker loop.

Real-Redis integration is gated behind ``SHEAF_INTEGRATION_REDIS=1`` so
CI runs against in-memory fakes; the integration test in
``test_smoke_worker.py`` exercises the Redis adapters end-to-end when
the env var is set.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any

import pytest

from sheaf.api.base import ModelType
from sheaf.worker import (
    Job,
    JobQueue,
    JobQueueClient,
    JobResult,
    ResultStore,
    SheafWorker,
    WorkerSpec,
)

# Register the smoke backends in this process.
os.environ.setdefault("SHEAF_EXTRA_BACKENDS", "tests.stubs")
import tests.stubs  # noqa: E402, F401

# ---------------------------------------------------------------------------
# In-memory fakes — no Redis required
# ---------------------------------------------------------------------------


class FakeQueue(JobQueue):
    """In-memory FIFO queue.  Tracks ack/nack/dead_letter for assertions."""

    def __init__(self) -> None:
        self._pending: deque[Job] = deque()
        self.acked: list[str] = []
        self.nacked: list[str] = []
        self.dead_lettered: list[tuple[str, str]] = []  # (job_id, reason)
        self._next_id = 0

    def enqueue(
        self,
        request: dict,
        webhook_url: str | None = None,
        job_id: str | None = None,
    ) -> str:
        self._next_id += 1
        msg_id = f"msg-{self._next_id}"
        jid = job_id or f"job-{self._next_id}"
        self._pending.append(
            Job(
                job_id=jid,
                msg_id=msg_id,
                request=request,
                webhook_url=webhook_url,
                submitted_at=0.0,
            )
        )
        return jid

    def dequeue(self, block_ms: int) -> Job | None:
        if not self._pending:
            return None
        return self._pending.popleft()

    def ack(self, job: Job) -> None:
        self.acked.append(job.job_id)

    def nack(self, job: Job) -> None:
        # Re-deliver: push back to the front so the next dequeue gets it.
        self.nacked.append(job.job_id)
        self._pending.appendleft(job)

    def dead_letter(self, job: Job, reason: str) -> None:
        self.dead_lettered.append((job.job_id, reason))


class FakeResultStore(ResultStore):
    def __init__(self) -> None:
        self._store: dict[str, JobResult] = {}

    def put(self, result: JobResult) -> None:
        self._store[result.job_id] = result

    def get(self, job_id: str) -> JobResult | None:
        return self._store.get(job_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def queue() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
def results() -> FakeResultStore:
    return FakeResultStore()


@pytest.fixture
def ts_request() -> dict:
    return {
        "model_type": "time_series",
        "model_name": "smoke",
        "history": [1.0, 2.0, 3.0, 4.0],
        "horizon": 3,
        "frequency": "1h",
    }


def _make_spec(queue: JobQueue, results: ResultStore, **overrides: Any) -> WorkerSpec:
    base: dict[str, Any] = {
        "name": "test-worker",
        "model_type": ModelType.TIME_SERIES,
        "backend": "_smoke_ts",
        "queue": queue,
        "results": results,
        "max_retries": 2,
        "poll_block_ms": 10,
    }
    base.update(overrides)
    return WorkerSpec(**base)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def test_spec_constructs(queue, results):
    spec = _make_spec(queue, results)
    assert spec.name == "test-worker"
    assert spec.max_retries == 2
    assert spec.queue is queue
    assert spec.results is results


def test_spec_rejects_negative_max_retries(queue, results):
    with pytest.raises(ValueError):
        _make_spec(queue, results, max_retries=-1)


def test_spec_rejects_nonpositive_poll_block_ms(queue, results):
    with pytest.raises(ValueError):
        _make_spec(queue, results, poll_block_ms=0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def test_client_enqueue_validates_request(queue, results):
    client = JobQueueClient(queue, results)
    with pytest.raises(Exception):  # ValidationError
        client.enqueue({"model_type": "time_series", "history": "not a list"})


def test_client_enqueue_returns_job_id(queue, results, ts_request):
    client = JobQueueClient(queue, results)
    jid = client.enqueue(ts_request)
    assert isinstance(jid, str) and jid.startswith("job-")


def test_client_wait_for_result_times_out(queue, results):
    client = JobQueueClient(queue, results)
    with pytest.raises(TimeoutError):
        client.wait_for_result("nonexistent", timeout_s=0.05, poll_interval_s=0.01)


def test_client_wait_for_result_returns(queue, results):
    client = JobQueueClient(queue, results)
    results.put(
        JobResult(job_id="abc", status="completed", response={"x": 1}, completed_at=0.0)
    )
    out = client.wait_for_result("abc", timeout_s=0.5)
    assert out.status == "completed"
    assert out.response == {"x": 1}


# ---------------------------------------------------------------------------
# Worker loop — happy path
# ---------------------------------------------------------------------------


def test_worker_processes_one_job(queue, results, ts_request):
    spec = _make_spec(queue, results)
    client = JobQueueClient(queue, results)
    jid = client.enqueue(ts_request)

    worker = SheafWorker(spec)
    job = worker.run_one()

    assert job is not None and job.job_id == jid
    assert queue.acked == [jid]
    out = results.get(jid)
    assert out is not None
    assert out.status == "completed"
    assert out.response is not None
    assert out.response["model_name"] == "smoke"
    assert out.response["model_type"] == "time_series"
    assert out.response["horizon"] == 3
    assert out.response["mean"] == [0.42, 0.42, 0.42]


def test_worker_run_one_returns_none_when_idle(queue, results):
    spec = _make_spec(queue, results)
    worker = SheafWorker(spec)
    assert worker.run_one() is None


# ---------------------------------------------------------------------------
# Worker loop — model_type mismatch
# ---------------------------------------------------------------------------


def test_worker_dead_letters_model_type_mismatch(queue, results, ts_request):
    # Backend is _smoke_ts (TIME_SERIES) but spec says TABULAR — every job
    # fails validation.  After max_retries+1 attempts, dead-letter.
    spec = _make_spec(
        queue,
        results,
        model_type=ModelType.TABULAR,
        backend="_smoke_tabular",
        max_retries=1,
    )
    client = JobQueueClient(queue, results)
    jid = client.enqueue(ts_request)

    worker = SheafWorker(spec)
    # Delivery 1, max_retries=1 → retries_remaining=0 → dead-letter immediately.
    worker.run_one()

    assert queue.dead_lettered and queue.dead_lettered[0][0] == jid
    out = results.get(jid)
    assert out is not None and out.status == "failed"
    assert "does not match" in (out.error or "")


# ---------------------------------------------------------------------------
# Worker loop — retry + dead-letter on backend errors
# ---------------------------------------------------------------------------


def test_worker_retries_then_dead_letters(queue, results, ts_request):
    spec = _make_spec(queue, results, backend="_smoke_error", max_retries=2)
    client = JobQueueClient(queue, results)
    jid = client.enqueue(ts_request)

    worker = SheafWorker(spec)
    # max_retries=2, delivery_count starts at 1.
    # delivery 1: retries_remaining = 2 - 1 = 1 → nack
    # delivery 2: retries_remaining = 2 - 2 = 0 → still nacks (1 > 0 is False... wait)
    # Code: retries_remaining = max_retries - delivery_count.  > 0 means retry.
    #   delivery 1 → 2-1=1 > 0 → nack
    #   delivery 2 → 2-2=0 not > 0 → dead-letter
    worker.run_one()
    assert queue.nacked == [jid]
    assert not queue.dead_lettered

    worker.run_one()
    assert queue.dead_lettered and queue.dead_lettered[0][0] == jid
    out = results.get(jid)
    assert out is not None and out.status == "failed"
    assert "backend exploded" in (out.error or "")


# ---------------------------------------------------------------------------
# Worker loop — webhook on completion
# ---------------------------------------------------------------------------


def test_worker_calls_webhook_on_completion(queue, results, ts_request, monkeypatch):
    spec = _make_spec(queue, results)
    client = JobQueueClient(queue, results)
    jid = client.enqueue(ts_request, webhook_url="https://example.test/cb")

    posted: list[dict[str, Any]] = []

    class _FakeHttpx:
        @staticmethod
        def post(url: str, json: dict, timeout: float) -> None:
            posted.append({"url": url, "body": json, "timeout": timeout})

    # Inject the fake httpx into the runner's import path.
    import sys

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)

    SheafWorker(spec).run_one()

    assert len(posted) == 1
    assert posted[0]["url"] == "https://example.test/cb"
    body = posted[0]["body"]
    assert body["job_id"] == jid
    assert body["status"] == "completed"
    assert body["error"] is None
    assert body["response"]["mean"] == [0.42, 0.42, 0.42]


def test_webhook_failure_does_not_break_worker(queue, results, ts_request, monkeypatch):
    """A webhook POST that raises must not prevent ack or fail the job."""
    spec = _make_spec(queue, results)
    client = JobQueueClient(queue, results)
    jid = client.enqueue(ts_request, webhook_url="https://broken.test/cb")

    class _ExplodingHttpx:
        @staticmethod
        def post(url: str, json: dict, timeout: float) -> None:
            raise RuntimeError("network down")

    import sys

    monkeypatch.setitem(sys.modules, "httpx", _ExplodingHttpx)

    SheafWorker(spec).run_one()

    # Result still persisted, queue still acked, no dead-letter.
    assert queue.acked == [jid]
    assert results.get(jid) is not None
    assert not queue.dead_lettered
