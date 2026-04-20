"""BatchRunner — offline batch inference via Ray Data map_batches.

Reuses every existing ``ModelBackend`` unchanged.  Execution is Ray Data
``map_batches`` in stateless task mode: the backend is instantiated and
loaded inside the worker process and cached at module level so ``load()``
fires once per worker rather than once per batch.

Deferred follow-ups:
  * Resumable checkpointing across process restarts — #12
  * Actor-pool execution mode for warm loads on models with expensive
    ``load()`` (FLUX, GraphCast, SDXL) — #13
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import TypeAdapter

from sheaf.api.union import AnyRequest
from sheaf.backends.base import ModelBackend
from sheaf.batch.spec import BatchSpec, JsonlSink, JsonlSource

_logger = logging.getLogger(__name__)

# Worker-local backend cache. Ray Data runs map_batches as stateless tasks
# but reuses worker processes across tasks within a single job, so loading
# the backend at module level means load() fires once per worker rather
# than once per batch. Keyed by spec.name so two jobs in the same process
# don't collide.
_BACKEND_CACHE: dict[str, ModelBackend] = {}

_REQUEST_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyRequest)


def _build_backend(spec: BatchSpec) -> ModelBackend:
    # Worker processes don't inherit the driver's import state, so populate
    # the backend registry on each worker before looking up the backend.
    # Idempotent; cheap when already imported.
    from sheaf.backends._register import (
        register_builtin_backends,
        register_extra_backends,
    )

    register_builtin_backends()
    register_extra_backends()

    # Re-import the live registry — the module-level import may be a stale
    # cloudpickle snapshot in worker processes.
    from sheaf.registry import _BACKEND_REGISTRY as _registry

    backend_cls = spec.backend_cls or _registry.get(spec.backend)
    if backend_cls is None:
        raise ValueError(
            f"Unknown backend '{spec.backend}'. Registered backends: {list(_registry)}"
        )
    backend: ModelBackend = backend_cls(**spec.backend_kwargs)
    backend.load()
    return backend


def _get_or_load_backend(spec: BatchSpec) -> ModelBackend:
    if spec.name not in _BACKEND_CACHE:
        _BACKEND_CACHE[spec.name] = _build_backend(spec)
    return _BACKEND_CACHE[spec.name]


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


class BatchRunner:
    """Offline batch inference over a ``BatchSpec`` source → sink pipeline.

    Example:
        spec = BatchSpec(
            name="chronos-batch",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={"model_size": "small"},
            source=JsonlSource(path="in.jsonl"),
            sink=JsonlSink(path="out.jsonl"),
            batch_size=64,
        )
        BatchRunner(spec).run()
    """

    def __init__(self, spec: BatchSpec) -> None:
        self._spec = spec

    def run(self) -> int:
        """Run the batch job end-to-end.  Returns the number of output rows."""
        try:
            import pandas as pd  # ty: ignore[unresolved-import]
            import pyarrow  # noqa: F401  # ty: ignore[unresolved-import]
            import ray  # ty: ignore[unresolved-import]
            import ray.data  # noqa: F401  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "Batch inference requires ray.data + pandas + pyarrow. "
                "Install with: pip install 'sheaf-serve[batch]'"
            ) from e

        spec = self._spec

        if not isinstance(spec.source, JsonlSource):
            raise TypeError(
                f"Unsupported source type {type(spec.source).__name__}. "
                "v0.6 supports JsonlSource only."
            )
        if not isinstance(spec.sink, JsonlSink):
            raise TypeError(
                f"Unsupported sink type {type(spec.sink).__name__}. "
                "v0.6 supports JsonlSink only."
            )

        rows = _read_jsonl(spec.source.path)
        _logger.info(
            "BatchRunner %s: loaded %d rows from %s",
            spec.name,
            len(rows),
            spec.source.path,
        )

        # Pre-validate in the driver so users see schema errors up-front
        # rather than halfway through a long distributed job.
        for i, row in enumerate(rows):
            try:
                req = _REQUEST_ADAPTER.validate_python(row)
            except Exception as exc:
                raise ValueError(f"row {i} failed validation: {exc}") from exc
            if req.model_type != spec.model_type:
                raise ValueError(
                    f"row {i} model_type={req.model_type!r} does not match "
                    f"BatchSpec.model_type={spec.model_type!r}"
                )

        if not ray.is_initialized():
            ray.init()

        ds = ray.data.from_items(rows)

        captured_spec = spec

        def _infer(pdf: pd.DataFrame) -> pd.DataFrame:
            backend = _get_or_load_backend(captured_spec)
            request_dicts = pdf.to_dict(orient="records")
            requests = [_REQUEST_ADAPTER.validate_python(r) for r in request_dicts]
            responses = backend.batch_predict(requests)
            return pd.DataFrame([r.model_dump(mode="json") for r in responses])

        result_ds = ds.map_batches(
            _infer,  # ty: ignore[invalid-argument-type]
            batch_format="pandas",
            batch_size=spec.batch_size,
            num_cpus=spec.num_cpus,
            num_gpus=spec.num_gpus,
        )

        # Ray Data preserves row order across map operations; take_all()
        # returns rows in input order.
        output_rows = result_ds.take_all()
        _write_jsonl(spec.sink.path, output_rows)
        _logger.info(
            "BatchRunner %s: wrote %d rows to %s",
            spec.name,
            len(output_rows),
            spec.sink.path,
        )
        return len(output_rows)
