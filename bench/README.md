# Sheaf benchmarks

A reproducible head-to-head against the obvious alternatives, on the same
machine, with the same model, the same workload, and the same batching
parameters. Numbers are checked into [`results/`](./results/) so anyone can
see what we measured and how we measured it.

## What this benchmark is for

The point is **not** "sheaf is faster." Sheaf wraps Ray Serve; on the
serving hot path it is mostly the same code. The point is to show that
**the typed-contract layer adds no measurable overhead** — and to put a
specific number on what sheaf gives you for the lines of code you write.

So we measure two things, deliberately:

1. **Latency under sustained load** — p50/p95/p99 at a fixed RPS. The
   honest claim is *parity*; if sheaf is materially slower than raw Ray
   Serve under the same configuration, that is a regression and we want
   to know.
2. **Lines of code per server** — the actual user-facing surface area
   needed to deploy the same model. This is where the framework-vs-no-
   framework comparison lives.

We don't claim throughput wins, p99 wins, or feature wins. The benchmark
exists to be honest, not to win.

## What we benchmark

- **Model**: `amazon/chronos-bolt-tiny` — ~80MB, runs on CPU, no GPU
  required, no auth tokens. Picked because it's the most accessible
  model in the sheaf catalog; results should generalise to any
  CPU-bound forecaster.
- **Workload**: 64-step history → 12-step horizon, `output_mode=mean`.
  Static input pre-built once, replayed by the load generator.
- **Frameworks**:
  - **sheaf-serve** ([`servers/sheaf_server.py`](./servers/sheaf_server.py))
  - **raw Ray Serve** ([`servers/ray_serve_server.py`](./servers/ray_serve_server.py))
  - **BentoML** ([`servers/bentoml_server.py`](./servers/bentoml_server.py))

All three use:
- Same model (`amazon/chronos-bolt-tiny`)
- Same batch size limit (`max_batch_size=8`)
- Same batch wait window (`10ms`)
- Single replica, single worker process

## Methodology

- **Open-loop load** — the [load generator](./loadgen.py) sends requests at
  a sustained target RPS, regardless of how long previous requests take.
  This avoids [coordinated omission](https://www.scylladb.com/2021/04/22/on-coordinated-omission/),
  which closed-loop generators (one-request-at-a-time-per-virtual-user)
  silently introduce. Closed-loop hides p99 tail because slow requests
  prevent new ones from being sent during the slow window.
- **Warm state** — every run includes a 10-second warmup at the target RPS
  before measurement starts. Cold-start latency is reported separately
  and is not mixed into the steady-state distribution.
- **Single machine** — server and load generator run on the same host so
  network latency is loopback-only. The bench is about the framework, not
  the network.
- **httpx client** — async, HTTP/1.1, connection pool sized to allow
  concurrent in-flight requests up to the configured limit.

## How to reproduce

```bash
# In repo root
uv sync --extra time-series  # for sheaf + ray-serve servers (chronos)
uv pip install bentoml         # only if comparing against bentoml

# Terminal 1 — start one of the servers (pick one):
uv run python bench/servers/sheaf_server.py
# or
uv run python bench/servers/ray_serve_server.py
# or
uv run --with bentoml python bench/servers/bentoml_server.py

# Terminal 2 — run the load generator
uv run python bench/loadgen.py --target-rps 50 --duration 30 --url http://127.0.0.1:8000/forecaster/predict
```

The load generator prints a results JSON on stdout; pipe it into
`bench/results/` to capture a run.

## Hardware

Each results file in [`results/`](./results/) records the hardware it was
run on (CPU model, core count, RAM, OS). Numbers are not portable across
machines — re-run the bench on your own to know what to expect on yours.

Latest run: [`results/2026-05-09-cpu-m1.md`](./results/2026-05-09-cpu-m1.md)
— Apple M1, 20 RPS, 20 s window. tl;dr: sheaf-serve is at parity with
raw Ray Serve (the same substrate); BentoML is faster than both at low
RPS because Ray Serve's multi-actor hop overhead dominates when most
batches contain one request. The framework-vs-no-framework SLOC delta
favours sheaf by ~60%.

## What the results tell you

If the tails (p99) are within ~10% of each other across all three
frameworks at the same RPS, sheaf is doing its job: the typed-contract
layer is free. If sheaf's tail blows out, that's a real regression and
we want a bug report.

The LoC table at the bottom of each results file is the other half of
the comparison: the same model, same batching, same protocol — sheaf
gets you there in fewer lines of *your* code.
