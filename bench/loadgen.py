"""Open-loop async HTTP load generator.

Sends requests at a sustained target RPS regardless of how long previous
requests take.  Avoids coordinated omission, which closed-loop generators
(one-request-at-a-time-per-virtual-user) silently introduce.

Usage:
    uv run python bench/loadgen.py \
        --url http://127.0.0.1:8000/forecaster/predict \
        --target-rps 50 \
        --duration 30 \
        --warmup 10

Reports a JSON results blob on stdout suitable for archiving in
``bench/results/``.

Honest measurement notes:
- Latency is wall-clock from request send to response received, including
  the loopback network hop (negligible) and the framework overhead
  (the thing we're measuring).
- p99 with N samples requires N >= 100 to be meaningful, N >= 1000 to be
  trustworthy.  At target_rps=50 with duration=30s that's 1500 samples,
  comfortable.
- The achieved RPS field reports what we actually sent; if the server is
  saturated and rejecting connections the reported RPS will be lower
  than the target — note that and don't lie about it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import httpx

# A single representative payload, replayed for every request.  Same
# shape across all three servers (sheaf, raw Ray Serve, BentoML); they
# all accept history + horizon.  64-step history → 12-step horizon is
# representative of intraday-load forecasting.
_PAYLOAD = {
    "model_type": "time_series",
    "model_name": "forecaster",
    "history": [100.0 + 5.0 * ((i * 7) % 13) for i in range(64)],
    "horizon": 12,
    "frequency": "1h",
    "output_mode": "mean",
}


@dataclass
class RunResult:
    target_rps: int
    duration_s: int
    warmup_s: int
    url: str
    achieved_rps: float
    n_samples: int
    n_errors: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    server: str = ""  # filled in by --label flag
    hardware: dict = field(default_factory=dict)


def _hw() -> dict:
    """Best-effort capture of what we're running on."""
    info: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        # uname -a-like single-line summary
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            info["cpu_model"] = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info


async def _one(
    client: httpx.AsyncClient, url: str, payload: dict
) -> tuple[float, bool]:
    """Send one request, return (latency_ms, ok)."""
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=30.0)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return (time.perf_counter() - t0) * 1000.0, ok


def _build_payload(wire: str) -> dict:
    """Adjust the wire payload per framework.

    Sheaf and raw Ray Serve accept a flat request body.  BentoML's
    batchable API with a ``list[Request]`` parameter exposes a wrapped
    wire contract — ``{"requests": [<payload>]}`` per call, even for
    one request.  Keeping batching on for BentoML (the fair
    comparison) means the loadgen has to wrap the body for that
    server.  The latency we measure is still per-call wall-clock; the
    wrapping itself is a single dict allocation.
    """
    if wire == "bentoml":
        return {"requests": [_PAYLOAD]}
    return _PAYLOAD


async def _run(args: argparse.Namespace) -> RunResult:
    interval = 1.0 / args.target_rps
    total_s = args.warmup + args.duration
    payload = _build_payload(args.wire)

    # Single shared connection pool large enough that the generator
    # itself isn't the bottleneck at high RPS.  100 sustains 100 RPS
    # against a 1s p99 without queueing on our side.
    limits = httpx.Limits(
        max_connections=200,
        max_keepalive_connections=200,
    )

    samples_warm: list[float] = []
    errors_warm = 0
    tasks: list[asyncio.Task] = []

    async with httpx.AsyncClient(limits=limits) as client:
        # Single sanity request first — fail fast if URL is wrong.
        latency_ms, ok = await _one(client, args.url, payload)
        if not ok:
            print(
                f"ERROR: pre-flight request to {args.url} failed.  "
                "Is the server up?  Did you set --url correctly for "
                "the framework you're benching?",
                file=sys.stderr,
            )
            sys.exit(2)

        start = time.perf_counter()
        next_fire = start
        end = start + total_s

        # Open-loop firing: we schedule a request every `interval` seconds
        # of wall-clock, regardless of whether the previous one has come
        # back.  This is the entire point of an open-loop generator.
        while True:
            now = time.perf_counter()
            if now >= end:
                break
            if now < next_fire:
                # Sleep until the next firing tick.  asyncio.sleep gives
                # millisecond granularity which is fine at our RPS.
                await asyncio.sleep(next_fire - now)

            measure_phase = (time.perf_counter() - start) >= args.warmup

            async def _track(t0: float, measure: bool) -> None:
                nonlocal errors_warm
                latency_ms, ok = await _one(client, args.url, payload)
                if measure:
                    if ok:
                        samples_warm.append(latency_ms)
                    else:
                        errors_warm += 1

            tasks.append(
                asyncio.create_task(_track(time.perf_counter(), measure_phase))
            )
            next_fire += interval

        # Drain in-flight requests.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    if not samples_warm:
        print(
            "ERROR: no successful samples in measurement window — server "
            "may be overloaded or refusing connections.  Lower --target-rps.",
            file=sys.stderr,
        )
        sys.exit(3)

    samples_warm.sort()
    n = len(samples_warm)

    return RunResult(
        target_rps=args.target_rps,
        duration_s=args.duration,
        warmup_s=args.warmup,
        url=args.url,
        achieved_rps=n / args.duration,
        n_samples=n,
        n_errors=errors_warm,
        p50_ms=statistics.median(samples_warm),
        p95_ms=samples_warm[int(0.95 * n)] if n >= 20 else samples_warm[-1],
        p99_ms=samples_warm[int(0.99 * n)] if n >= 100 else samples_warm[-1],
        mean_ms=statistics.fmean(samples_warm),
        max_ms=samples_warm[-1],
        server=args.label,
        hardware=_hw(),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="POST endpoint to bench")
    p.add_argument(
        "--target-rps",
        type=int,
        default=50,
        help="Target sustained RPS (open-loop)",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Measurement window in seconds, after warmup",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup window in seconds (samples discarded)",
    )
    p.add_argument(
        "--label",
        default="",
        help=(
            "Server label written into the result "
            "(e.g. 'sheaf', 'ray-serve', 'bentoml')"
        ),
    )
    p.add_argument(
        "--wire",
        default="flat",
        choices=["flat", "bentoml"],
        help=(
            "Wire format: 'flat' for sheaf + raw Ray Serve, 'bentoml' wraps "
            "the payload in {'requests': [...]} for BentoML's batchable API."
        ),
    )
    args = p.parse_args()

    result = asyncio.run(_run(args))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
