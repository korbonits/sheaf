"""Offline batch inference quickstart — JSONL in → JSONL out.

Demonstrates ``BatchRunner`` running Chronos-Bolt over a JSONL file with
Ray Data as the execution substrate.  The same backend used by Ray Serve
(`backends.chronos.Chronos2Backend`) is reused unchanged — batch mode just
re-homes it onto ``ds.map_batches`` and writes results to a sink file.

Key points:
  - ``BatchSpec`` declares the job: which backend, which source/sink, how
    many rows per ``batch_predict`` call, CPU/GPU reservation per Ray Data
    task.
  - ``BatchRunner.run()`` pre-validates every input row against the typed
    Pydantic contract *on the driver* before dispatching, so schema errors
    surface up-front (not halfway through a long distributed run).
  - Output row order matches input row order — Ray Data preserves lineage
    across ``map_batches``.
  - Two follow-up tracks are deferred to issues:
      * Resumable checkpointing across process restarts — #12
      * Actor-pool mode for warm loads on expensive ``load()`` — #13

Usage::

    pip install 'sheaf-serve[time-series,batch]'
    uv run --extra batch --extra time-series python examples/quickstart_batch.py

The ``--extra batch`` flag is required under ``uv run`` so Ray Data workers
inherit pandas + pyarrow; plain ``python`` with an activated venv also works.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sheaf.api.base import ModelType
from sheaf.batch import BatchRunner, BatchSpec, JsonlSink, JsonlSource

# ---------------------------------------------------------------------------
# 1. Write a JSONL input file — one TimeSeriesRequest dict per line.
# ---------------------------------------------------------------------------

tmp = Path(tempfile.mkdtemp(prefix="sheaf-batch-"))
input_path = tmp / "requests.jsonl"
output_path = tmp / "forecasts.jsonl"

series = [
    # Hourly electricity demand, 3 zones
    [312, 298, 275, 260, 255, 263, 285, 320, 368, 402, 421, 435],
    [180, 172, 165, 160, 158, 162, 175, 195, 220, 245, 262, 275],
    [95, 92, 88, 84, 82, 85, 94, 108, 125, 140, 150, 158],
    # Daily sales, 2 SKUs
    [42, 55, 48, 61, 58, 63, 70, 65, 72, 80, 78, 85],
    [120, 135, 128, 142, 150, 145, 160, 155, 168, 175, 170, 185],
]

requests = [
    {
        "model_type": "time_series",
        "model_name": f"zone-{i:02d}" if i < 3 else f"sku-{i - 3:02d}",
        "history": h,
        "horizon": 6,
        "frequency": "1h" if i < 3 else "1d",
        "output_mode": "mean",
    }
    for i, h in enumerate(series)
]

input_path.write_text("\n".join(json.dumps(r) for r in requests))
print(f"Wrote {len(requests)} requests → {input_path}")

# ---------------------------------------------------------------------------
# 2. Build a BatchSpec.
# ---------------------------------------------------------------------------

spec = BatchSpec(
    name="chronos-batch",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
        "torch_dtype": "float32",
    },
    source=JsonlSource(path=str(input_path)),
    sink=JsonlSink(path=str(output_path)),
    batch_size=3,  # 3 rows per map_batches call; 5 inputs → 2 batches
    num_cpus=1.0,
    num_gpus=0.0,
)

# ---------------------------------------------------------------------------
# 3. Run.  First invocation downloads ~80MB for chronos-bolt-tiny.
# ---------------------------------------------------------------------------

print(f"Running BatchRunner (batch_size={spec.batch_size})…")
n = BatchRunner(spec).run()
print(f"Wrote {n} rows → {output_path}\n")

# ---------------------------------------------------------------------------
# 4. Inspect outputs.  Row order matches input row order.
# ---------------------------------------------------------------------------

print(f"{'model_name':<12}  {'horizon':>7}  mean")
print("-" * 70)
with open(output_path) as f:
    for line in f:
        row = json.loads(line)
        mean = [round(x, 1) for x in row["mean"]]
        print(f"{row['model_name']:<12}  {row['horizon']:>7}  {mean}")

print(f"\nTemp dir: {tmp}")
