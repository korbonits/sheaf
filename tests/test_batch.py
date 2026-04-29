"""Tests for sheaf.batch — BatchSpec, JSONL I/O, and end-to-end BatchRunner."""

from __future__ import annotations

import json
import os

import pytest

from sheaf.api.base import ModelType
from sheaf.batch import BatchRunner, BatchSpec, JsonlSink, JsonlSource
from sheaf.batch.runner import _BACKEND_CACHE, _read_jsonl, _write_jsonl

# Ensure the smoke backends register in both driver and workers.
os.environ["SHEAF_EXTRA_BACKENDS"] = "tests.stubs"
import tests.stubs  # noqa: E402, F401

# Skip the e2e tests if pandas/pyarrow aren't installed (i.e., [batch] extra
# wasn't synced). Lint and unit tests still run.
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("ray.data")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_worker_backend_cache() -> None:
    """Clear the worker-local backend cache between tests so state doesn't leak."""
    _BACKEND_CACHE.clear()


@pytest.fixture
def ts_rows() -> list[dict]:
    return [
        {
            "model_type": "time_series",
            "model_name": "smoke",
            "history": [1.0, 2.0, 3.0, 4.0],
            "horizon": 3,
            "frequency": "1h",
        },
        {
            "model_type": "time_series",
            "model_name": "smoke",
            "history": [10.0, 20.0],
            "horizon": 2,
            "frequency": "1d",
        },
        {
            "model_type": "time_series",
            "model_name": "smoke",
            "history": [7.0, 8.0, 9.0],
            "horizon": 4,
            "frequency": "1h",
        },
    ]


# ---------------------------------------------------------------------------
# Unit: JSONL helpers
# ---------------------------------------------------------------------------


def test_read_write_jsonl_round_trip(tmp_path):
    path = str(tmp_path / "rt.jsonl")
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    _write_jsonl(path, rows)
    assert _read_jsonl(path) == rows


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = str(tmp_path / "blanks.jsonl")
    path_obj = tmp_path / "blanks.jsonl"
    path_obj.write_text('{"a": 1}\n\n{"a": 2}\n   \n')
    assert _read_jsonl(path) == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# Unit: BatchSpec
# ---------------------------------------------------------------------------


def test_batch_spec_construction(tmp_path):
    spec = BatchSpec(
        name="t",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(tmp_path / "in.jsonl")),
        sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
    )
    assert spec.batch_size == 32
    assert spec.num_cpus == 1.0
    assert spec.num_gpus == 0.0
    assert isinstance(spec.source, JsonlSource)
    assert isinstance(spec.sink, JsonlSink)


def test_batch_spec_rejects_nonpositive_batch_size(tmp_path):
    with pytest.raises(ValueError):
        BatchSpec(
            name="t",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            source=JsonlSource(path=str(tmp_path / "in.jsonl")),
            sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
            batch_size=0,
        )


def test_batch_spec_actors_requires_num_actors(tmp_path):
    with pytest.raises(ValueError, match="num_actors is required"):
        BatchSpec(
            name="t",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            source=JsonlSource(path=str(tmp_path / "in.jsonl")),
            sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
            compute="actors",
        )


def test_batch_spec_actors_with_num_actors(tmp_path):
    spec = BatchSpec(
        name="t",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(tmp_path / "in.jsonl")),
        sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
        compute="actors",
        num_actors=2,
    )
    assert spec.compute == "actors"
    assert spec.num_actors == 2


def test_batch_spec_rejects_nonpositive_num_actors(tmp_path):
    with pytest.raises(ValueError):
        BatchSpec(
            name="t",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            source=JsonlSource(path=str(tmp_path / "in.jsonl")),
            sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
            compute="actors",
            num_actors=0,
        )


# ---------------------------------------------------------------------------
# Driver validation — must happen before Ray dispatch
# ---------------------------------------------------------------------------


def test_validation_rejects_bad_row(tmp_path, ts_rows):
    src = tmp_path / "bad.jsonl"
    bad = list(ts_rows)
    # Invalid frequency — Pydantic enum validation should reject.
    bad.append({**ts_rows[0], "frequency": "yearly"})
    src.write_text("\n".join(json.dumps(r) for r in bad))

    spec = BatchSpec(
        name="bad",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
    )
    with pytest.raises(ValueError, match="row 3 failed validation"):
        BatchRunner(spec).run()


def test_validation_rejects_model_type_mismatch(tmp_path, ts_rows):
    src = tmp_path / "mt.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in ts_rows))

    spec = BatchSpec(
        name="mt",
        model_type=ModelType.TABULAR,  # mismatch — rows are TIME_SERIES
        backend="_smoke_tabular",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
    )
    with pytest.raises(ValueError, match="does not match"):
        BatchRunner(spec).run()


def test_rejects_unsupported_source_type(tmp_path):
    class _WeirdSource(JsonlSource):
        pass

    # Construct normally then monkey-patch the source to a non-Jsonl subclass
    # of BatchSource to simulate a future source type the runner doesn't grok.
    from sheaf.batch.spec import BatchSource

    class _FutureSource(BatchSource):
        uri: str

    spec = BatchSpec(
        name="f",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=_FutureSource(uri="s3://bucket/key"),
        sink=JsonlSink(path=str(tmp_path / "out.jsonl")),
    )
    with pytest.raises(TypeError, match="Unsupported source"):
        BatchRunner(spec).run()


# ---------------------------------------------------------------------------
# End-to-end — JSONL in, Ray Data map_batches, JSONL out
# ---------------------------------------------------------------------------


def test_end_to_end_time_series(tmp_path, ts_rows):
    src = tmp_path / "in.jsonl"
    sink = tmp_path / "out.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in ts_rows))

    spec = BatchSpec(
        name="e2e-ts",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(sink)),
        batch_size=2,
    )
    n = BatchRunner(spec).run()

    assert n == len(ts_rows)
    output = _read_jsonl(str(sink))
    assert len(output) == len(ts_rows)

    # _smoke_ts returns mean = [0.42] * horizon for each request.
    for row_in, row_out in zip(ts_rows, output):
        assert row_out["horizon"] == row_in["horizon"]
        assert row_out["mean"] == [0.42] * row_in["horizon"]


def test_end_to_end_actor_pool(tmp_path, ts_rows):
    """End-to-end with compute='actors' — load() runs once per actor."""
    src = tmp_path / "in.jsonl"
    sink = tmp_path / "out.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in ts_rows))

    spec = BatchSpec(
        name="e2e-ts-actors",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(sink)),
        batch_size=2,
        compute="actors",
        num_actors=1,
    )
    n = BatchRunner(spec).run()

    assert n == len(ts_rows)
    output = _read_jsonl(str(sink))
    assert len(output) == len(ts_rows)
    for row_in, row_out in zip(ts_rows, output):
        assert row_out["horizon"] == row_in["horizon"]
        assert row_out["mean"] == [0.42] * row_in["horizon"]


def test_end_to_end_actor_pool_multi_actor_preserves_order(tmp_path):
    """Multi-actor dispatch must preserve input row order — Ray Data lineage."""
    rows = [
        {
            "model_type": "time_series",
            "model_name": f"req-{i:03d}",
            "history": [float(i)] * 5,
            "horizon": 2,
            "frequency": "1h",
        }
        for i in range(25)
    ]
    src = tmp_path / "ordered_actors.jsonl"
    sink = tmp_path / "ordered_actors_out.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))

    spec = BatchSpec(
        name="order-actors",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(sink)),
        batch_size=4,
        compute="actors",
        num_actors=3,
    )
    BatchRunner(spec).run()

    output = _read_jsonl(str(sink))
    assert [r["model_name"] for r in output] == [r["model_name"] for r in rows]


def test_end_to_end_actor_pool_propagates_backend_errors(tmp_path):
    """Exceptions raised inside _BackendActor.__call__ must reach the driver."""
    rows = [
        {
            "model_type": "time_series",
            "model_name": "boom",
            "history": [1.0, 2.0, 3.0],
            "horizon": 2,
            "frequency": "1h",
        }
    ]
    src = tmp_path / "err.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))

    spec = BatchSpec(
        name="err-actors",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_error",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(tmp_path / "err_out.jsonl")),
        compute="actors",
        num_actors=1,
    )
    with pytest.raises(Exception, match="backend exploded"):
        BatchRunner(spec).run()


def test_end_to_end_preserves_input_order(tmp_path):
    # Give each row a distinct model_name so we can verify order-preservation
    # even when batch_size splits rows into multiple batches.
    rows = [
        {
            "model_type": "time_series",
            "model_name": f"req-{i:03d}",
            "history": [float(i)] * 3,
            "horizon": 2,
            "frequency": "1h",
        }
        for i in range(7)
    ]
    src = tmp_path / "ordered.jsonl"
    sink = tmp_path / "ordered_out.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))

    spec = BatchSpec(
        name="order",
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        source=JsonlSource(path=str(src)),
        sink=JsonlSink(path=str(sink)),
        batch_size=3,
    )
    BatchRunner(spec).run()

    output = _read_jsonl(str(sink))
    assert [r["model_name"] for r in output] == [r["model_name"] for r in rows]
