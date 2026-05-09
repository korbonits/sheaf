# Offline batch jobs

For workloads where you don't need a live HTTP path — overnight
forecasts for 100k entities, embedding a full image catalogue,
back-filling predictions across a year of history — sheaf-serve ships
`BatchRunner`. It reads JSONL, runs any registered backend over the
rows via Ray Data's `map_batches`, and writes JSONL out.

```bash
pip install "sheaf-serve[batch,time-series]"
```

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.batch import BatchRunner, BatchSpec

spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={"model_id": "amazon/chronos-bolt-tiny"},
)

runner = BatchRunner(model_spec=spec)
runner.run(BatchSpec(
    source="s3://my-bucket/inputs.jsonl",
    sink="s3://my-bucket/outputs.jsonl",
    batch_size=64,
))
```

The input file holds one request per line — JSON matching the
backend's request schema (here `TimeSeriesRequest`). The output file
holds one response per line, in the same order as the inputs.

## Two execution modes

### `compute="tasks"` (default)

Stateless Ray Data tasks. A module-level `_BACKEND_CACHE: dict[str,
ModelBackend]` keyed by `spec.name` means `backend.load()` fires once
per worker process (not once per batch). Best for:

- Cheap loads (small models)
- Short jobs where actor startup overhead would dominate
- Variable-load scenarios where Ray can scale tasks elastically

### `compute="actors"`

Warm `_BackendActor` actors with a fixed pool size. The backend is
loaded once per actor at construction time and held for the actor's
lifetime. Best for:

- Expensive `load()` (FLUX, SDXL, GraphCast — minutes to load)
- Large jobs where the cold-start cost amortises
- GPU-bound work where you want predictable replica residency

```python
runner.run(BatchSpec(
    source="...",
    sink="...",
    batch_size=8,
    compute="actors",
    num_actors=4,   # 4 GPUs → 4 actors, one per GPU
))
```

Sizing constraint: `num_actors * spec.resources.num_gpus` must be
`<= cluster GPUs`. The cold-start cost (first batch per actor blocks
on `load()`) is the price of the warm-load guarantee — pre-warm with
a dummy batch if it matters.

## Determinism — row order

Ray Data's streaming executor (the default since Ray 2.x) does **not**
preserve input order across `map_batches`. `BatchRunner` injects a
`_sheaf_row_idx` sentinel column on `from_items`, propagates it
through the per-batch handler, then sorts and drops it after
`take_all()`. Output order matches input order under both `tasks` and
`actors` modes — relied upon by downstream joins.

## Driver-side validation

Rows are validated against `TypeAdapter(AnyRequest)` and their
`model_type` checked against `spec.model_type` on the driver,
**before** `ray.data.from_items()`. Schema errors surface up front,
not halfway through a long distributed job.

## Reference

Full schema in the [Batch runner API reference](../api-reference/batch.md).
