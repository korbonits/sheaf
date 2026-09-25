"""ESM C kit on Modal H100: fetch pinned weights, prove exact == off, benchmark.

Builds the ESM C kit's pinned stack as a Modal image — a step-for-step mirror
of the kit's own ``esmc/environment/Dockerfile`` (Ubuntu 24.04 + CUDA 13.0.1
runtime, Python 3.12, ``requirements.lock`` installed ``--no-deps``, nvcc +
ninja, ``run.sh install``) — then layers Sheaf's in-repo ``src/`` on top
without changing any pinned package (the kit's pin check runs last).

    modal run bench/model_opt/modal_esmc.py::fetch_weights   # once
    modal run bench/model_opt/modal_esmc.py::gpu_test        # exact == off, bitwise
    modal run bench/model_opt/modal_esmc.py::bench           # off vs exact numbers

Volumes: ``sheaf-esmc-kit-weights`` holds the digest-checked Hugging Face
snapshot (``/weights/esmc/hf``, the kit's layout); ``sheaf-model-opt-jit``
holds the compile caches (``/jit``, ``MODEL_OPT_JIT_ROOT``) so later
containers skip the first-run compiles.  The kit's CUDA extension is built
into the image at build time (``/root/.cache/esmc_sdkfused``).

Kits tree: ``SHEAF_KITS_REPO`` / ``SHEAF_KITS_REF`` (defaults below) — point
them at the Sheaf-maintained fork once it exists; the ref is a full commit
SHA so the image is reproducible.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

KITS_REPO = os.environ.get(
    "SHEAF_KITS_REPO",
    "https://github.com/anthropics/uplifting-biomolecular-modeling.git",
)
KITS_REF = os.environ.get("SHEAF_KITS_REF", "f4f62fa6592ae4938d49b1757bea0cfeff9f468e")
VARIANT = os.environ.get("SHEAF_ESMC_VARIANT", "6b")
GPU = os.environ.get("SHEAF_BENCH_GPU", "H100")
MINUTES = 60

_WEIGHTS = "/weights"
_JIT = "/jit"
_SHEAF = "/root/sheaf"
# Modal imports this file as /root/modal_esmc.py in the container.
_REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(_SHEAF)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


app = modal.App("sheaf-esmc-kit")
weights_vol = modal.Volume.from_name("sheaf-esmc-kit-weights", create_if_missing=True)
jit_vol = modal.Volume.from_name("sheaf-model-opt-jit", create_if_missing=True)

image = (
    # The kit Dockerfile's base: Ubuntu 24.04 (libstdc++ with CXXABI_1.3.15,
    # which the pinned flash-attn / TE wheels need) + the CUDA 13.0.1 runtime.
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-runtime-ubuntu24.04", add_python="3.12"
    )
    .env(
        {
            "PYTHONHASHSEED": "0",
            "CFLAGS": "-g0",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:"
            "/usr/local/cuda/lib64",
            "HF_HUB_OFFLINE": "1",
        }
    )
    .apt_install("ca-certificates", "curl", "git", "build-essential", "clang")
    # Only the ESM C kit's directory of the (large) kits tree.
    .run_commands(
        f"git clone --filter=blob:none --no-checkout {KITS_REPO} /kits",
        "git -C /kits sparse-checkout set --no-cone /esmc /LICENSE /NOTICE /README.md",
        f"git -C /kits checkout {KITS_REF}",
    )
    # The pinned stack, exactly the lock, no resolver (kit Dockerfile step 2).
    .run_commands(
        "echo setuptools==84.0.0 > /tmp/build-constraints.txt"
        " && PIP_CONSTRAINT=/tmp/build-constraints.txt python -m pip install"
        " --no-cache-dir --no-deps -r /kits/esmc/environment/requirements.lock"
    )
    .apt_install("cuda-nvcc-13-0", "ninja-build")
    # Kit editable + pin check + CUDA extension (sm80 + sm90: no GPU at build).
    .run_commands("cd /kits/esmc && bash run.sh install")
    # Sheaf runs from src/ (PYTHONPATH); its runtime needs (pydantic, httpx,
    # numpy) are already pinned by the lock.  Prove the pins still hold.
    .run_commands("python -I /kits/esmc/stock/check_pins.py")
    .env(
        {
            "PYTHONPATH": f"{_SHEAF}/src:{_SHEAF}",
            "HF_HOME": f"{_WEIGHTS}/esmc/hf",
            "SHEAF_KITS_REPO": KITS_REPO,
            "SHEAF_KITS_REF": KITS_REF,
            "SHEAF_GIT_SHA": _git_sha(),
        }
    )
    .add_local_dir(_REPO / "src", f"{_SHEAF}/src")
    .add_local_dir(_REPO / "tests", f"{_SHEAF}/tests")
    .add_local_dir(_REPO / "bench", f"{_SHEAF}/bench")
)

_volumes = {_WEIGHTS: weights_vol, _JIT: jit_vol}


def _fetch_pinned_snapshot(variant: str) -> None:
    """Download the kit's pinned snapshot by commit and point refs/main at it.

    The kit's downloader fetches the repo's default revision and refuses when
    upstream has re-published since the pin (biohub/ESMC-6B was re-sharded on
    2026-09-14).  With the pinned files already in place, ``run.sh install``
    skips its download and only runs its sha256 check.  refs/main must name
    the pinned commit because offline loads resolve through it.
    """
    import json

    os.environ["HF_HUB_OFFLINE"] = "0"  # read when huggingface_hub is imported
    from huggingface_hub import scan_cache_dir, snapshot_download

    pins = json.loads(Path("/kits/esmc/stock/PINS.json").read_text())
    repo = pins["variants"][variant]["hf_repo"]
    commit = pins["weights"][repo]["snapshot_commit"]
    cache = f"{_WEIGHTS}/esmc/hf/hub"
    snapshot_download(
        repo,
        revision=commit,
        cache_dir=cache,
        allow_patterns=["*.json", "*.safetensors"],
    )
    ref = Path(cache) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(exist_ok=True)  # snapshot_download(revision=<sha>) writes no refs/
    ref.write_text(commit)
    info = scan_cache_dir(cache)
    stale = [
        rev.commit_hash
        for cached in info.repos
        if cached.repo_id == repo
        for rev in cached.revisions
        if rev.commit_hash != commit
    ]
    if stale:
        info.delete_revisions(*stale).execute()


@app.function(image=image, volumes=_volumes, timeout=60 * MINUTES)
def fetch_weights(variant: str = VARIANT) -> None:
    """Fetch the kit's pinned snapshot and sha256-check every file."""
    _fetch_pinned_snapshot(variant)
    subprocess.run(
        [
            "bash",
            "run.sh",
            "install",
            "--weights",
            f"{_WEIGHTS}/esmc",
            "--variant",
            variant,
        ],
        cwd="/kits/esmc",
        check=True,
    )
    weights_vol.commit()


@app.function(image=image, volumes=_volumes, gpu=GPU, timeout=60 * MINUTES)
def gpu_test() -> str:
    """tests/test_gpu_model_opt_esmc.py: exact vs off, bit for bit."""
    env = dict(os.environ, SHEAF_GPU_KIT_TEST="1", MODEL_OPT_JIT_ROOT=_JIT)
    env["SHEAF_GPU_KIT_ESMC_MODEL"] = f"esmc_{VARIANT}"
    proc = subprocess.run(
        ["python", "-m", "pytest", "-v", "-rP", "tests/test_gpu_model_opt_esmc.py"],
        cwd=_SHEAF,
        env=env,
        capture_output=True,
        text=True,
    )
    jit_vol.commit()
    print(proc.stdout, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU equivalence test failed (rc={proc.returncode})")
    return proc.stdout


@app.function(image=image, volumes=_volumes, gpu=GPU, timeout=120 * MINUTES)
def run_bench(extra_args: list[str]) -> dict[str, str]:
    out = "/tmp/esmc-bench"
    subprocess.run(
        [
            "python",
            "bench/model_opt/bench_esmc.py",
            "--model",
            f"esmc_{VARIANT}",
            "--jit-root",
            _JIT,
            "--out-dir",
            out,
            *extra_args,
        ],
        cwd=_SHEAF,
        check=True,
    )
    jit_vol.commit()
    return {p.name: p.read_text() for p in Path(out).iterdir()}


@app.local_entrypoint()
def bench(args: str = "") -> None:
    """Run the benchmark on Modal and write results under bench/results/."""
    files = run_bench.remote(args.split())
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    dest = _REPO / "bench" / "results" / f"{stamp}-esmc-kit-{GPU.lower()}-{VARIANT}"
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (dest / name).write_text(text)
    print(f"wrote {dest}")
    print(files.get("README.md", ""))
