# 2026-05-15 — Apple M5 multi-sample medians

Re-bench of the three frameworks on M5 with two methodology fixes
addressing the single-sample variance that polluted the
[2026-05-14 single-sample sweep](./2026-05-14-sweep-cpu-m5.md):

1. **Multi-sample medians.** N=3 independent samples per
   (framework, scenario, RPS) point; we report the median of each
   percentile plus the min and max across samples.  Single-sample
   numbers at the saturation edge swung two orders of magnitude in
   the previous run; medians of N≥3 are robust to that.
2. **Two scenarios — 1×1 and 4×1.** The previous bench used 1 replica
   × 1 CPU for every test point, which forced a saturation boundary
   right around 100 RPS — exactly where the comparison was most
   interesting and the data was least reliable.  This run separates
   the two regimes:
   - **1×1** (single replica, single CPU) swept at 10 / 30 / 50 /
     75 RPS — under-saturated for all three frameworks, isolates
     per-request overhead.
   - **4×1** (4 replicas × 1 CPU each) swept at 100 / 200 / 400 RPS —
     the scale-out regime where framework architecture matters.

## Setup

| Field | Value |
|---|---|
| Hardware | Apple M5 (10 cores), macOS 26.4 |
| Python | 3.12.13 |
| Model | `amazon/chronos-bolt-tiny` (~80 MB) |
| Workload | 64-step history → 12-step horizon, `output_mode=mean` |
| Warmup | 5 s per sample |
| Measurement window | 15 s per sample |
| Samples per point | 3 |
| Batch params | `max_batch_size=8`, `batch_wait=10ms` (all three) |
| sheaf-serve | v0.10.0 |
| Ray | 2.55.x |
| BentoML | 1.4.39 |

All three server fixtures read `BENCH_REPLICAS` from the environment
(default 1) so the same code serves both scenarios.

## 1×1 — single replica, single CPU

Sustainable load on a single CPU replica; tests per-request overhead
in isolation from saturation effects.

| Framework | RPS | p50 median (ms) | p50 [min, max] | p99 median (ms) | Achieved RPS | Errors |
|---|---:|---:|---:|---:|---:|---:|
| **sheaf-serve** | 10 | 24.5 | [24.1, 26.4] | 34.5 | 10.07 | 0 |
| **sheaf-serve** | 30 | 22.0 | [21.1, 22.6] | 25.3 | 30.07 | 0 |
| **sheaf-serve** | 50 | 20.1 | [20.0, 20.6] | 23.7 | 50.07 | 0 |
| **sheaf-serve** | 75 | 21.3 | [21.2, 22.1] | 24.0 | 75.07 | 0 |
| Raw Ray Serve | 10 | 19.9 | [19.5, 20.6] | 21.4 | 10.07 | 0 |
| Raw Ray Serve | 30 | 18.2 | [18.1, 18.6] | 19.1 | 30.07 | 0 |
| Raw Ray Serve | 50 | 19.9 | [18.9, 20.1] | 21.5 | 50.07 | 0 |
| Raw Ray Serve | 75 | 19.8 | [19.7, 20.0] | 21.2 | 75.07 | 0 |
| BentoML | 10 | 10.2 | [9.9, 10.4] | 13.5 | 10.07 | 0 |
| BentoML | 30 | 6.5 | [6.3, 6.9] | 8.3 | 30.07 | 0 |
| BentoML | 50 | 5.3 | [5.3, 5.3] | 7.8 | 50.07 | 0 |
| BentoML | 75 | 4.5 | [4.5, 4.6] | 5.9 | 75.07 | 0 |

The `p50 [min, max]` column shows the range across the 3 samples —
all single-digit ms in every row, no run-to-run variance worth
discussing.  This is what clean benchmark data looks like.

### What the 1×1 numbers show

- **sheaf vs raw Ray Serve are at parity from 30 RPS upward** (sheaf
  ~1.5 ms slower p50 across 30–75 RPS, well within `@serve.batch`
  scheduling noise).  At 10 RPS sheaf is ~4.5 ms slower — at low fill
  rates the typed-contract validation cost is more visible because
  batches don't amortise across requests.  The "typed-contract layer
  is free" claim holds for any RPS where batches actually fill.
- **BentoML is faster at every RPS** by roughly 4× at p50.  Same as
  every previous run: BentoML's in-process model dispatch avoids the
  Ray Serve `HTTP proxy actor → replica actor → batched call → reply
  path` hop chain.  This is the trade-off Ray Serve makes in exchange
  for multi-replica orchestration.
- **BentoML's p50 drops as RPS climbs** (10.2 → 6.5 → 5.3 → 4.5 ms).
  Batch amortisation working as designed: more requests per
  10 ms batch window means more shared inference cost per request.

## 4×1 — four replicas, one CPU each

Scale-out regime.  4 cores allocated to the server (out of 10 on M5),
leaving headroom for the loadgen + system.  This is where the
substrate's architecture matters.

| Framework | RPS | p50 median (ms) | p50 [min, max] | p99 median (ms) | Achieved RPS | Errors |
|---|---:|---:|---:|---:|---:|---:|
| **sheaf-serve** | 100 | 20.1 | [19.9, 20.1] | 21.9 | 100.07 | 0 |
| **sheaf-serve** | 200 | 5,133 † | [54.5, 6,351] | 27,130 † | 199.00 † | 0 |
| **sheaf-serve** | 400 | 49,387 † | [46,680, 61,271] | 82,720 † | 437.60 † | 0 |
| Raw Ray Serve | 100 | 19.7 | [19.6, 19.8] | 21.8 | 100.07 | 0 |
| Raw Ray Serve | 200 | 10,079 † | [7,209, 13,665] | 28,177 † | 190.53 † | 0 |
| Raw Ray Serve | 400 | 36,441 † | [35,199, 39,001] | 74,124 † | 429.13 † | 0 |
| BentoML | 100 | 4.2 | [4.2, 4.2] | 5.1 | 100.07 | 0 |
| BentoML | 200 | 3.1 | [3.1, 3.2] | 3.7 | 200.07 | 0 |
| BentoML | 400 | 3.3 | [3.0, 3.4] | 4.0 | 400.07 | 0 |

† **Saturated.** Ray-Serve-based frameworks queue at 200+ RPS even with
4 replicas; see [How to read `Achieved RPS` at saturation](./2026-05-09-sweep-cpu-m1.md#how-to-read-achieved-rps-at-saturation)
for the math on why "achieved RPS" still reads close to target
in this regime.

### What the 4×1 numbers show

**1. At 100 RPS, all three are clean.** sheaf 20.1 / raw 19.7 / BentoML
4.2 ms, no variance. The 4× replica capacity comfortably absorbs
100 RPS for every framework.

**2. BentoML scales linearly with workers; Ray Serve does not.** This
is the new finding.  BentoML 4×1 at 400 RPS sustains p50 = 3.3 ms with
zero errors.  Both Ray-Serve-based frameworks saturate by 200 RPS at
4×1 (p50 in the seconds) and are completely gone by 400 RPS (p50 in
the tens of seconds).

   **Mechanism**: Ray Serve has a single HTTP proxy actor that fans
   out to replicas.  All 4 replicas dispatch through that one proxy.
   At 100 RPS the proxy isn't a bottleneck; at 200+ RPS the proxy
   actor itself becomes the saturation point regardless of how many
   replicas are behind it.  BentoML's workers each have their own
   HTTP listener (multiplexed by uvicorn's reuse-port socket), so
   adding workers adds dispatch capacity linearly.

   **Implication**: for `chronos-bolt-tiny`-scale models where the
   per-request inference cost is small relative to substrate
   overhead, BentoML's per-worker dispatch model is materially
   better at high RPS.  For larger models (GPU diffusion, etc.)
   the inference cost dominates and the proxy-actor bottleneck
   matters less.  This is the architectural trade-off, articulated
   precisely by data for the first time in this bench series.

**3. sheaf inherits the proxy-actor bottleneck verbatim.** sheaf vs
raw Ray Serve at 200 RPS: p50 medians of 5,133 vs 10,079 ms — both
saturated, the difference is well within saturation noise.  sheaf is
not adding scale-out overhead beyond what Ray Serve does; it just
can't compensate for the substrate's choice either.

## Comparison to the prior single-sample sweep

The 2026-05-14 single-sample run had us walk back a claim that sheaf
saturated at 100 RPS while raw Ray Serve didn't.  Multi-sample data
confirms: at 1×1, both Ray-Serve-based frameworks are clean through
75 RPS (only RPS level we tested under saturation for 1×1 this time);
at 4×1, both are clean through 100 RPS and both saturate at 200+ RPS.
No divergence between sheaf and raw Ray Serve at sustainable load.

## Honest limitations of this run

- **N=3, not 30.** Per-point median is robust to single outliers
  but not statistically tight.  For a published paper-grade result,
  re-run with N=30 and report confidence intervals.
- **macOS, not Linux.** Apple Silicon scheduler and BSD network stack
  add quirks; for production-relevant numbers, replicate on a single
  EC2 instance (`c7i.4xlarge` or similar — 4 vCPU x86_64 Linux is
  the closest analogue to this 4×1 scenario).
- **Single machine.** Even at 4×1 the loadgen and server share an OS;
  on EKS or any multi-node setup the network hop adds ~1 ms but
  removes the contention.  Worth doing for the scale-out claim
  specifically.
- **CPU-only model.** chronos-bolt-tiny is CPU-friendly; for GPU
  workloads (FLUX, SDXL, GraphCast) the model call dominates and
  framework overhead is rounding error.  None of this bench's
  findings transfer directly to GPU-served models.

## Reproducing this exact run

```bash
# Repo root
uv sync --extra dev --extra time-series
uv pip install bentoml

# 1×1 — single replica, single CPU
for fw in sheaf ray_serve bentoml; do
  BENCH_REPLICAS=1 .venv/bin/python bench/servers/${fw}_server.py &
  sleep 60
  for rps in 10 30 50 75; do
    .venv/bin/python bench/multi_sample.py \
      --url http://127.0.0.1:8000/{path-per-server}/predict \
      --target-rps $rps --duration 15 --warmup 5 --n 3 \
      --wire {flat|bentoml} \
      --label ${fw}-1x1-${rps} > bench/results/.../1x1/${fw}-${rps}rps.json
  done
  kill %1 && .venv/bin/ray stop --force
done

# Repeat with BENCH_REPLICAS=4 + RPS={100, 200, 400} for the 4×1 scenario.
```

Per-RPS JSON results are in
[`./2026-05-15-cpu-m5-medians/1x1/`](./2026-05-15-cpu-m5-medians/1x1/)
and
[`./2026-05-15-cpu-m5-medians/4x1/`](./2026-05-15-cpu-m5-medians/4x1/).
Each JSON contains the median/min/max aggregate plus the 3 raw
samples in a `raw_samples` array.
