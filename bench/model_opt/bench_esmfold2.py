"""Benchmark ESMFold2Backend under the ESMFold2 kit's modes: off vs exact vs fast.

One mode per process (the kit patches the interpreter process-wide), so the
driver re-invokes this script once per mode and merges the results:

    python bench/model_opt/bench_esmfold2.py --out-dir results/esmfold2-h100
    python bench/model_opt/bench_esmfold2.py --one-mode fast --json fast.json

``off`` is the library's fused backend with chunking off (what ``exact``
reproduces).  For each sequence length it reports, per mode, the p50 / p95
wall time of ``backend.predict`` (one fold plus response serialisation, which
is small next to the fold) after one warm-up fold at that length (CUDA graphs
are captured per input shape), and peak GPU memory.  Fold settings default to
the library's own (20 loops, 200 sampling steps, one sample), where the kit
documents its gains; Sheaf's request defaults are lighter (3 / 50), so pass
``--num-loops 3 --num-sampling-steps 50`` to measure those.

Sequences are deterministic (seeded, 20 standard amino acids).  The
environment block is shared with ``bench_esmc.py``, plus ``esmfold2_opt``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_esmc import _dist_version, _pct, _sequences, environment  # noqa: E402

MODES = ("off", "exact", "fast")
DEFAULT_LENGTHS = (128, 256, 512, 768)


def run_one_mode(args: argparse.Namespace) -> dict[str, Any]:
    import torch  # ty: ignore[unresolved-import]

    from sheaf.api.structure import ChainInput, StructureRequest
    from sheaf.backends.esmfold2 import ESMFold2Backend
    from sheaf.model_opt import ModelOptConfig, apply_model_opt

    backend = ESMFold2Backend(model_name=args.model, device="cuda")
    apply_model_opt(
        backend,
        ModelOptConfig(mode=args.one_mode, jit_root=args.jit_root),
        f"bench-{args.one_mode}",
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    backend.load()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    cells = []
    for length in args.lengths:
        (seq,) = _sequences(length, 1, args.seed)
        req = StructureRequest(
            model_name="esmfold2",
            chains=[ChainInput(chain_id="A", sequence=seq)],
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            seed=args.seed,
        )
        cell: dict[str, Any] = {"length": length}
        try:
            for _ in range(args.warmup):
                backend.predict(req)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times: list[float] = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t = time.perf_counter()
                backend.predict(req)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            cell.update(
                predict_s_p50=statistics.median(times),
                predict_s_p95=_pct(times, 0.95),
                predict_s_all=times,
                peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        except torch.cuda.OutOfMemoryError:
            cell["status"] = "oom"
            torch.cuda.empty_cache()
        else:
            cell["status"] = "ok"
        print(json.dumps({k: v for k, v in cell.items() if k != "predict_s_all"}))
        cells.append(cell)

    env = environment()
    env.pop("esmc_opt", None)
    env["esmfold2_opt"] = _dist_version("esmfold2_opt")
    return {
        "mode": args.one_mode,
        "model": args.model,
        "load_s": load_s,
        "kit_lines": backend.model_opt_lines,
        "env": env,
        "config": {
            "lengths": args.lengths,
            "num_loops": args.num_loops,
            "num_sampling_steps": args.num_sampling_steps,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "cells": cells,
    }


def render_markdown(results: dict[str, dict[str, Any]]) -> str:
    base = results["off"]
    env = base["env"]
    cfg = base["config"]
    kit_env = next((r["env"] for m, r in results.items() if m != "off"), env)
    lines = [
        f"# ESMFold2 kit benchmark — {base['model']}",
        "",
        f"- GPU: {env.get('gpu')} (cc {env.get('compute_capability')}, "
        f"{env.get('gpu_mem_mib')} MiB); nvidia-smi: {env.get('nvidia_smi')}",
        f"- torch {env.get('torch')} / CUDA {env.get('cuda_runtime')} / cuDNN "
        f"{env.get('cudnn')} / triton {env.get('triton')}",
        f"- flash-attn {env.get('flash_attn')} / Transformer Engine "
        f"{env.get('transformer_engine')} / Python {env.get('python')}",
        f"- esm {env.get('esm')} ({env.get('esm_source')})",
        f"- transformers {env.get('transformers')} ({env.get('transformers_source')})",
        f"- esmfold2_opt {kit_env.get('esmfold2_opt')}; "
        f"kits {env.get('kits_repo')}@{env.get('kits_ref')}",
        f"- sheaf {env.get('sheaf')} @ {env.get('sheaf_git_sha')}",
        f"- measured {env.get('timestamp_utc')}; num_loops={cfg['num_loops']}, "
        f"num_sampling_steps={cfg['num_sampling_steps']}, one sample; "
        f"{cfg['iters']} timed folds per length ({cfg['warmup']} warm-up)",
        "",
        "Load time: "
        + ", ".join(f"{m} {r['load_s']:.1f} s" for m, r in results.items()),
        "",
        "`off` = the library's fused backend with chunking off, the configuration "
        "`exact` reproduces bit for bit. Time = `backend.predict` wall (one fold "
        "+ response serialisation). Speedup = off p50 / mode p50.",
        "",
        "| L | mode | predict p50 s | p95 s | speedup | peak alloc MiB |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    by_mode = {m: {c["length"]: c for c in r["cells"]} for m, r in results.items()}
    for length, c0 in by_mode["off"].items():
        for mode in results:
            c = by_mode[mode].get(length, {})
            if c.get("status") != "ok":
                lines.append(f"| {length} | {mode} | {c.get('status')} | | | |")
                continue
            sp = (
                f"{c0['predict_s_p50'] / c['predict_s_p50']:.2f}x"
                if c0.get("status") == "ok"
                else "—"
            )
            lines.append(
                f"| {length} | {mode} | {c['predict_s_p50']:.2f} "
                f"| {c['predict_s_p95']:.2f} | {sp} | {c['peak_alloc_mib']:,.0f} |"
            )
    for mode in results:
        if mode != "off":
            lines += ["", f"Kit lines ({mode}):", ""]
            lines += [f"    {ln}" for ln in results[mode].get("kit_lines", [])]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="biohub/ESMFold2")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    ap.add_argument("--num-loops", type=int, default=20)
    ap.add_argument("--num-sampling-steps", type=int, default=200)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jit-root", default=os.environ.get("MODEL_OPT_JIT_ROOT"))
    ap.add_argument("--out-dir", default="bench/results/esmfold2-kit")
    # Internal: run a single mode in this process and write its JSON.
    ap.add_argument("--one-mode", choices=MODES)
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.one_mode:
        res = run_one_mode(args)
        Path(args.json).write_text(json.dumps(res, indent=1))
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        path = out / f"{mode}.json"
        cmd = [sys.executable, __file__, "--one-mode", mode, "--json", str(path)]
        flags = ("model", "num_loops", "num_sampling_steps", "iters", "warmup", "seed")
        for flag in flags:
            cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        cmd += ["--lengths", *map(str, args.lengths)]
        if args.jit_root:
            cmd += ["--jit-root", args.jit_root]
        env = dict(os.environ)
        env.pop("ESMFOLD2_OPT", None)  # the explicit enable() path only
        print(f"== mode {mode}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, env=env)
        results[mode] = json.loads(path.read_text())
    if "off" in results:
        (out / "README.md").write_text(render_markdown(results))
        print((out / "README.md").read_text())


if __name__ == "__main__":
    main()
