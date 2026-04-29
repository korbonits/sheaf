"""Offline batch inference with an actor pool — warm `load()` per actor.

Same JSONL → JSONL pipeline as ``quickstart_batch.py``, but with
``compute="actors"``: an actor pool of size ``num_actors`` is created up
front; each actor calls ``backend.load()`` once at ``__init__`` and the
loaded model persists for the actor's lifetime.

Use this mode when ``load()`` is expensive — FLUX (~30-60s),
GraphCast / SDXL (~20-40s), large Whisper, etc. — and the per-batch cost
of the stateless task path's worker-cache fallback would dominate the
total job time.

Trade-offs vs ``compute="tasks"``:
  - First batch per actor blocks on ``load()``.  For FLUX that's minutes
    of cold start.  Pre-warm with a dummy batch if it matters.
  - GPUs are reserved for each actor's lifetime, not just per batch.
    Sizing constraint: ``num_actors * num_gpus <= cluster GPUs``.
  - No worker-local module cache (``_BACKEND_CACHE``) is consulted; the
    actor instance owns the loaded backend directly.

Usage::

    pip install 'sheaf-serve[time-series,batch]'
    uv run --extra batch --extra time-series \
        python examples/quickstart_batch_actors.py

The ``--extra batch`` flag is required under ``uv run`` so Ray Data
workers inherit pandas + pyarrow.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from sheaf.api.base import ModelType
from sheaf.batch import BatchRunner, BatchSpec, JsonlSink, JsonlSource

# ---------------------------------------------------------------------------
# 1. Build a JSONL input large enough to exercise multi-actor dispatch.
#    With num_actors=2 and batch_size=4, 12 rows splits into 3 batches that
#    Ray Data assigns across both actors.
# ---------------------------------------------------------------------------

tmp = Path(tempfile.mkdtemp(prefix="sheaf-batch-actors-"))
input_path = tmp / "requests.jsonl"
output_path = tmp / "forecasts.jsonl"

# 12 hourly demand series (small, synthetic — would normally be S3 / DB rows).
series = [
    [100 + i + 5 * (j % 4) for j in range(12)]  # gentle trend + low-freq seasonality
    for i in range(12)
]

requests = [
    {
        "model_type": "time_series",
        "model_name": f"series-{i:02d}",
        "history": h,
        "horizon": 6,
        "frequency": "1h",
        "output_mode": "mean",
    }
    for i, h in enumerate(series)
]

input_path.write_text("\n".join(json.dumps(r) for r in requests))
print(f"Wrote {len(requests)} requests → {input_path}")

# ---------------------------------------------------------------------------
# 2. BatchSpec with compute="actors".  num_actors is REQUIRED in actor mode;
#    the BatchSpec model_validator rejects a bare compute="actors".
# ---------------------------------------------------------------------------

spec = BatchSpec(
    name="chronos-batch-actors",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
        "torch_dtype": "float32",
    },
    source=JsonlSource(path=str(input_path)),
    sink=JsonlSink(path=str(output_path)),
    batch_size=4,
    num_cpus=1.0,  # per actor (not per task) in actor mode
    num_gpus=0.0,  # per actor; on a GPU box, set to 1.0 and size num_actors accordingly
    compute="actors",
    num_actors=2,
)

# ---------------------------------------------------------------------------
# 3. Run.  First batch per actor pays the load() cost (~few seconds for
#    chronos-bolt-tiny on CPU; minutes for FLUX on GPU).  Subsequent batches
#    on the same actor reuse the warm model.
# ---------------------------------------------------------------------------

print(
    f"Running BatchRunner (compute={spec.compute!r}, "
    f"num_actors={spec.num_actors}, batch_size={spec.batch_size})…"
)
t0 = time.time()
n = BatchRunner(spec).run()
dt = time.time() - t0
print(f"Wrote {n} rows → {output_path} in {dt:.2f}s\n")

# ---------------------------------------------------------------------------
# 4. Inspect outputs.  Order is preserved across actors — Ray Data lineage
#    tracking guarantees take_all() returns rows in input order even when
#    different batches were processed by different actors.
# ---------------------------------------------------------------------------

print(f"{'model_name':<14}  {'horizon':>7}  mean")
print("-" * 70)
with open(output_path) as f:
    for line in f:
        row = json.loads(line)
        mean = [round(x, 1) for x in row["mean"]]
        print(f"{row['model_name']:<14}  {row['horizon']:>7}  {mean}")

print(f"\nTemp dir: {tmp}")
