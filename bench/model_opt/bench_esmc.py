"""Benchmark ESMCBackend under the ESM C kit's modes: off vs exact.

One mode per process (the kit patches the interpreter process-wide), so the
driver re-invokes this script once per mode and merges the results:

    python bench/model_opt/bench_esmc.py --out-dir results/esmc-h100
    python bench/model_opt/bench_esmc.py --one-mode exact --json exact.json

For each (sequence length L, batch size B) cell it reports, per mode:

- ``forward_ms``  p50 / p95 of the model call alone (tokenize + forward under
  the backend's own autocast, ``torch.cuda.synchronize`` before and after) —
  where the kit acts;
- ``predict_ms``  p50 / p95 of ``backend.predict`` end to end, including
  response serialisation (logits → JSON-able lists), which is identical in
  both modes and dilutes the speedup — reported so nobody mistakes the
  forward speedup for the served one;
- ``residues_per_s`` = B * L / forward p50;
- ``peak_alloc_mib`` / ``peak_reserved_mib`` from ``torch.cuda`` peak stats
  over the cell.

Load time (``ESMC.from_pretrained`` + the kit's warm-up) is reported per mode.
Sequences are deterministic (seeded, 20 standard amino acids).  The
environment block records GPU model, driver, CUDA, torch, triton, flash-attn,
Transformer Engine, esm (+ commit), esmc_opt, the kits tree ref and Sheaf's
git SHA, so the numbers can be reproduced.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

MODES = ("off", "exact")
DEFAULT_LENGTHS = (64, 128, 256, 512, 1024, 2048)
DEFAULT_BATCHES = (1, 8, 32)
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def _sequences(length: int, batch: int, seed: int) -> list[str]:
    rng = random.Random(seed * 1_000_003 + length * 101 + batch)
    return ["".join(rng.choice(AMINO) for _ in range(length)) for _ in range(batch)]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _direct_url(name: str) -> str | None:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    info = json.loads(raw)
    commit = (info.get("vcs_info") or {}).get("commit_id")
    return f"{info.get('url')}@{commit}" if commit else info.get("url")


def environment() -> dict[str, Any]:
    import torch  # ty: ignore[unresolved-import]

    env: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": _dist_version("triton"),
        "flash_attn": _dist_version("flash_attn") or _dist_version("flash-attn"),
        "transformer_engine": _dist_version("transformer_engine"),
        "transformers": _dist_version("transformers"),
        "transformers_source": _direct_url("transformers"),
        "esm": _dist_version("esm"),
        "esm_source": _direct_url("esm"),
        "esmc_opt": _dist_version("esmc_opt"),
        "kits_repo": os.environ.get("SHEAF_KITS_REPO"),
        "kits_ref": os.environ.get("SHEAF_KITS_REF"),
        "sheaf": _dist_version("sheaf-serve"),
        "sheaf_git_sha": os.environ.get("SHEAF_GIT_SHA") or _git_sha(),
    }
    try:
        q = "name,driver_version,memory.total,clocks.max.sm,power.limit"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        env["nvidia_smi"] = out.splitlines()
    except (OSError, subprocess.CalledProcessError) as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = p.name
        env["gpu_mem_mib"] = p.total_memory // 2**20
        env["compute_capability"] = f"{p.major}.{p.minor}"
    return env


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_one_mode(args: argparse.Namespace) -> dict[str, Any]:
    import torch  # ty: ignore[unresolved-import]

    from sheaf.api.protein_language import ProteinLanguageRequest
    from sheaf.backends.esmc import ESMCBackend
    from sheaf.model_opt import ModelOptConfig, apply_model_opt

    backend = ESMCBackend(model_name=args.model, device="cuda")
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

    def forward(seqs: list[str]) -> None:
        # The model call of ESMCBackend._run, without response serialisation.
        enc = backend._tokenizer(seqs, return_tensors="pt", padding=True)
        inputs = {k: enc[k].to("cuda") for k in ("input_ids", "attention_mask")}
        with torch.inference_mode(), backend._autocast(torch):
            backend._model(**inputs, output_hidden_states=False)

    cells = []
    for length in args.lengths:
        for batch in args.batches:
            seqs = _sequences(length, batch, args.seed)
            cell: dict[str, Any] = {"length": length, "batch": batch}
            try:
                for _ in range(args.warmup):
                    forward(seqs)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                fwd: list[float] = []
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    forward(seqs)
                    torch.cuda.synchronize()
                    fwd.append((time.perf_counter() - t) * 1e3)
                pred: list[float] = []
                req = ProteinLanguageRequest(
                    model_name="esmc", sequences=seqs, return_logits=True
                )
                for _ in range(args.predict_iters):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    backend.predict(req)
                    torch.cuda.synchronize()
                    pred.append((time.perf_counter() - t) * 1e3)
                p50 = statistics.median(fwd)
                cell.update(
                    forward_ms_p50=p50,
                    forward_ms_p95=_pct(fwd, 0.95),
                    forward_ms_all=fwd,
                    predict_ms_p50=statistics.median(pred) if pred else None,
                    predict_ms_p95=_pct(pred, 0.95) if pred else None,
                    residues_per_s=batch * length / (p50 / 1e3),
                    peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                )
            except torch.cuda.OutOfMemoryError:
                cell["status"] = "oom"
                torch.cuda.empty_cache()
            else:
                cell["status"] = "ok"
            print(json.dumps({k: v for k, v in cell.items() if k != "forward_ms_all"}))
            cells.append(cell)

    return {
        "mode": args.one_mode,
        "model": args.model,
        "load_s": load_s,
        "kit_lines": backend.model_opt_lines,
        "env": environment(),
        "config": {
            "lengths": args.lengths,
            "batches": args.batches,
            "iters": args.iters,
            "warmup": args.warmup,
            "predict_iters": args.predict_iters,
            "seed": args.seed,
        },
        "cells": cells,
    }


def render_markdown(results: dict[str, dict[str, Any]]) -> str:
    base = results["off"]
    env = base["env"]
    lines = [
        f"# ESM C kit benchmark — {base['model']}",
        "",
        f"- GPU: {env.get('gpu')} (cc {env.get('compute_capability')}, "
        f"{env.get('gpu_mem_mib')} MiB); nvidia-smi: {env.get('nvidia_smi')}",
        f"- torch {env.get('torch')} / CUDA {env.get('cuda_runtime')} / cuDNN "
        f"{env.get('cudnn')} / triton {env.get('triton')}",
        f"- flash-attn {env.get('flash_attn')} / Transformer Engine "
        f"{env.get('transformer_engine')} / Python {env.get('python')}",
        f"- esm {env.get('esm')} ({env.get('esm_source')})",
        f"- transformers {env.get('transformers')} ({env.get('transformers_source')})",
        f"- esmc_opt {results.get('exact', {}).get('env', {}).get('esmc_opt')}; "
        f"kits {env.get('kits_repo')}@{env.get('kits_ref')}",
        f"- sheaf {env.get('sheaf')} @ {env.get('sheaf_git_sha')}",
        f"- measured {env.get('timestamp_utc')}; "
        f"{base['config']['iters']} timed iterations per cell "
        f"({base['config']['warmup']} warm-up)",
        "",
        "Load time: "
        + ", ".join(f"{m} {r['load_s']:.1f} s" for m, r in results.items()),
        "",
        "`forward` = model call only (where the kit acts). `predict` = "
        "`backend.predict` end to end, incl. JSON serialisation of logits "
        "(same cost in every mode). Speedup = off p50 / mode p50.",
        "",
        "| L | B | mode | forward p50 ms | p95 ms | speedup | residues/s "
        "| predict p50 ms | predict speedup | peak alloc MiB |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_mode = {
        m: {(c["length"], c["batch"]): c for c in r["cells"]}
        for m, r in results.items()
    }
    for key, c0 in by_mode["off"].items():
        for mode in results:
            c = by_mode[mode].get(key, {})
            if c.get("status") != "ok":
                lines.append(
                    f"| {key[0]} | {key[1]} | {mode} | {c.get('status')} | | | | | | |"
                )
                continue
            sp = (
                f"{c0['forward_ms_p50'] / c['forward_ms_p50']:.2f}x"
                if c0.get("status") == "ok"
                else "—"
            )
            psp = (
                f"{c0['predict_ms_p50'] / c['predict_ms_p50']:.2f}x"
                if c0.get("status") == "ok" and c.get("predict_ms_p50")
                else "—"
            )
            lines.append(
                f"| {key[0]} | {key[1]} | {mode} | {c['forward_ms_p50']:.2f} "
                f"| {c['forward_ms_p95']:.2f} | {sp} | {c['residues_per_s']:,.0f} "
                f"| {c['predict_ms_p50']:.1f} | {psp} | {c['peak_alloc_mib']:,.0f} |"
            )
    lines += ["", "Kit lines (exact):", ""]
    lines += [f"    {ln}" for ln in results.get("exact", {}).get("kit_lines", [])]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Biohub/ESMC-6B")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    ap.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--predict-iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jit-root", default=os.environ.get("MODEL_OPT_JIT_ROOT"))
    ap.add_argument("--out-dir", default="bench/results/esmc-kit")
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
        for flag in ("model", "iters", "warmup", "predict_iters", "seed"):
            cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        cmd += ["--lengths", *map(str, args.lengths)]
        cmd += ["--batches", *map(str, args.batches)]
        if args.jit_root:
            cmd += ["--jit-root", args.jit_root]
        env = dict(os.environ)
        env.pop("ESMC_OPT", None)  # the explicit enable() path only
        print(f"== mode {mode}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, env=env)
        results[mode] = json.loads(path.read_text())
    if "off" in results:
        (out / "README.md").write_text(render_markdown(results))
        print((out / "README.md").read_text())


if __name__ == "__main__":
    main()
