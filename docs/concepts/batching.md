# Batching

Each `ModelSpec` has a `BatchPolicy`. By default sheaf-serve uses
`@serve.batch` with `max_batch_size=32` and a `10 ms` wait window.
Tune both to fit your model's throughput-vs-latency curve.

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.scheduling.batch import BatchPolicy

spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    batch_policy=BatchPolicy(
        max_batch_size=64,
        timeout_ms=20,
    ),
)
```

`max_batch_size` and `timeout_ms` map directly onto Ray Serve's
[`@serve.batch`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.batch.html)
parameters; the deployment configures itself at boot via the per-batch
runtime setters.

## `bucket_by` — length-variable inputs

For inputs that vary in length within a batch (time-series histories of
different horizons, video clips with different frame counts), padding
the whole batch to the longest item wastes compute. `bucket_by` names
a scalar field on the request; requests with the same value share a
sub-batch, requests with different values go through separate
`batch_predict` calls in the same window.

```python
spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    batch_policy=BatchPolicy(
        max_batch_size=64,
        timeout_ms=20,
        bucket_by="horizon",   # group by forecast horizon
    ),
)
```

Within one Ray Serve batch window, sheaf-serve calls
`bucket_requests(requests, "horizon")` and dispatches one
`batch_predict` per bucket — preserving original arrival order so
results map back cleanly. Requests missing the field land in the
`None` bucket together.

The `ModalServer` path handles requests one at a time (no
`@serve.batch`), so `bucket_by` is silently ignored there.

## Adapter-aware sub-batching (LoRA)

When `ModelSpec.lora` is set, the bucket-by-resolved-adapter step
happens automatically inside each batch window. `pipeline.set_adapters`
on diffusers is process-global state, so two requests in the same
batch with different adapter selections must dispatch separately —
the deployment groups them transparently and calls
`set_active_adapters` once per group. See [LoRA multiplexing](lora.md)
for the full design.

`bucket_by` and `ModelSpec.lora` are mutually exclusive in v1; the
spec validator rejects the combination.

## When the defaults are wrong

| Symptom | What to change |
|---|---|
| Latency dominated by `timeout_ms` (small load, single requests) | Lower `timeout_ms`; or accept the latency cost in exchange for batching when load grows |
| Batch never fills (low RPS, large `max_batch_size`) | Lower `max_batch_size` or `timeout_ms`; the wait window is the latency floor for solo requests |
| OOM at high load | Lower `max_batch_size`; profile peak GPU memory at the cap |
| Mix of short + long sequences padding everything to max | Add `bucket_by="<length-field>"` |

## Reference

Full schema in the [Scheduling API reference](../api-reference/scheduling.md).
