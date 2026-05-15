"""Multi-sample wrapper around loadgen.py.

Runs ``bench/loadgen.py`` ``--n`` times against the given URL/RPS, then
emits a single JSON blob to stdout containing every sample plus median
and IQR for the headline latency percentiles.  Used by the sweep
driver to dampen single-sample variance at the saturation edge.

Usage::

    .venv/bin/python bench/multi_sample.py \\
        --url http://127.0.0.1:8000/forecaster/predict \\
        --target-rps 100 --duration 15 --warmup 5 \\
        --n 5 --label sheaf-4x1 --wire flat
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _median_iqr(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": float("nan"), "iqr": float("nan"), "min": float("nan"), "max": float("nan")}
    sorted_values = sorted(values)
    n = len(sorted_values)
    median = statistics.median(sorted_values)
    if n >= 4:
        q1 = statistics.median(sorted_values[: n // 2])
        q3 = statistics.median(sorted_values[(n + 1) // 2 :])
        iqr = q3 - q1
    else:
        iqr = float("nan")
    return {
        "median": median,
        "iqr": iqr,
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--target-rps", type=int, required=True)
    p.add_argument("--duration", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--n", type=int, default=5, help="number of independent samples")
    p.add_argument("--label", default="run")
    p.add_argument("--wire", default="flat", choices=["flat", "bentoml"])
    args = p.parse_args()

    loadgen = Path(__file__).parent / "loadgen.py"
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"

    samples: list[dict] = []
    for i in range(args.n):
        proc = subprocess.run(
            [
                str(venv_python),
                str(loadgen),
                "--url",
                args.url,
                "--target-rps",
                str(args.target_rps),
                "--duration",
                str(args.duration),
                "--warmup",
                str(args.warmup),
                "--wire",
                args.wire,
                "--label",
                f"{args.label}-{i}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        sample = json.loads(proc.stdout)
        samples.append(sample)
        print(
            f"  sample {i + 1}/{args.n}: p50={sample['p50_ms']:.1f} "
            f"p95={sample['p95_ms']:.1f} p99={sample['p99_ms']:.1f} "
            f"achieved={sample['achieved_rps']:.2f}",
            file=sys.stderr,
        )

    aggregate = {
        "label": args.label,
        "target_rps": args.target_rps,
        "duration_s": args.duration,
        "warmup_s": args.warmup,
        "n_samples": args.n,
        "url": args.url,
        "p50_ms": _median_iqr([s["p50_ms"] for s in samples]),
        "p95_ms": _median_iqr([s["p95_ms"] for s in samples]),
        "p99_ms": _median_iqr([s["p99_ms"] for s in samples]),
        "achieved_rps": _median_iqr([s["achieved_rps"] for s in samples]),
        "n_errors_total": sum(s["n_errors"] for s in samples),
        "raw_samples": samples,
    }
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
