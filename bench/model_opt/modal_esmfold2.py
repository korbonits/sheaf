"""ESMFold2 kit on Modal H100: fetch pinned weights, prove exact == off, benchmark.

Builds the ESMFold2 kit's pinned stack as a Modal image, following the kit's
own ``esmfold2/environment/Dockerfile``: Python 3.12.10 on Debian bookworm,
``requirements.lock`` installed ``--no-deps``, the CUDA 13.0 toolkit, the three
CUDA extensions (xformers, Transformer Engine, flash-attn) compiled by the
kit's ``environment/build_wheels.sh`` for compute capability 9.0, then
``run.sh install`` (kit + shared core editable, pin check).  The extension
build is the long step (~16 min on 24 cores); it runs as a ``run_function``
step so it gets 32 CPUs and 128 GiB, and is cached after the first build.

    modal run bench/model_opt/modal_esmfold2.py::fetch_weights  # once
    modal run bench/model_opt/modal_esmfold2.py::gpu_test  # exact == off; fast on
    modal run bench/model_opt/modal_esmfold2.py::bench  # off vs exact vs fast

Volumes: ``sheaf-esmfold2-kit-weights`` holds the three pinned snapshots
(``biohub/ESMFold2``, ``biohub/ESMFold2-Fast``, ``biohub/ESMC-6B``, ~27 GB) in
the kit's ``HF_HOME`` layout; the kit's installer fetches each file at its
pinned commit and points ``refs/main`` at it.  ``sheaf-model-opt-jit`` is the
compile cache shared with the ESM C runner (keyed by torch / CUDA / arch).
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
MODEL = os.environ.get("SHEAF_ESMFOLD2_MODEL", "biohub/ESMFold2")
GPU = os.environ.get("SHEAF_BENCH_GPU", "H100")
MINUTES = 60

_WEIGHTS = "/weights"
_HF_HOME = f"{_WEIGHTS}/esmfold2"
_JIT = "/jit"
_SHEAF = "/root/sheaf"
_KIT = "/kits/esmfold2"
_CUDA_HOME = "/usr/local/cuda-13.0"
# Modal imports this file as /root/modal_esmfold2.py in the container.
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


def _build_wheels() -> None:
    """The kit's own extension build (checks its prerequisites by name)."""
    subprocess.run(
        ["bash", "environment/build_wheels.sh", "--stack", "img_ef2_fa"],
        cwd=_KIT,
        env=dict(os.environ, CUDA_HOME=_CUDA_HOME),
        check=True,
    )


app = modal.App("sheaf-esmfold2-kit")
weights_vol = modal.Volume.from_name(
    "sheaf-esmfold2-kit-weights", create_if_missing=True
)
jit_vol = modal.Volume.from_name("sheaf-model-opt-jit", create_if_missing=True)

image = (
    modal.Image.from_registry("python:3.12.10-slim-bookworm")
    .env(
        {
            "PYTHONHASHSEED": "0",
            "CFLAGS": "-g0",
            "NVTE_FRAMEWORK": "pytorch",
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
        }
    )
    .apt_install("gcc", "gfortran", "build-essential", "git", "ca-certificates", "curl")
    # The ESMFold2 kit and the shared core it binds (common/opt_core).
    .run_commands(
        f"git clone --filter=blob:none --no-checkout {KITS_REPO} /kits",
        "git -C /kits sparse-checkout set --no-cone"
        " /esmfold2 /common/opt_core /LICENSE /NOTICE /README.md",
        f"git -C /kits checkout {KITS_REF}",
    )
    # The pinned stack, exactly the lock, no resolver; the three extensions
    # come from build_wheels.sh below (kit Dockerfile step 2).
    .run_commands(
        f"grep -E '^pip==' {_KIT}/environment/requirements.lock"
        " | xargs python -m pip install --no-cache-dir --no-deps",
        f"grep -v -E '^(#|(flash_attn|transformer_engine|xformers)==)'"
        f" {_KIT}/environment/requirements.lock > /tmp/stack.txt",
        "python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt",
    )
    # The CUDA 13.0 toolkit from NVIDIA's Debian 12 repository (keyring
    # digest from the kit's Dockerfile).
    .run_commands(
        "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com"
        "/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "echo 'e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d"
        "  /tmp/cuda-keyring.deb' | sha256sum -c -",
        "dpkg -i /tmp/cuda-keyring.deb && apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends"
        " cuda-compiler-13-0 cuda-libraries-dev-13-0 cuda-cudart-dev-13-0"
        " cuda-nvtx-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0",
    )
    .run_function(_build_wheels, cpu=32.0, memory=128 * 1024, timeout=180 * MINUTES)
    .run_commands(f"cd {_KIT} && bash run.sh install")
    # The GPU test's runner, which the kit's lock does not carry; --no-deps so
    # nothing pinned moves, then the kit's install re-runs its pin check.
    .run_commands(
        "python -m pip install --no-cache-dir --no-deps"
        " pytest==9.1.1 iniconfig==2.3.0 pluggy==1.6.0",
        f"cd {_KIT} && bash run.sh install",
    )
    .env(
        {
            "PYTHONPATH": f"{_SHEAF}/src:{_SHEAF}",
            "HF_HOME": _HF_HOME,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "MODEL_OPT_TARGET_GPU": GPU,
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


@app.function(image=image, volumes=_volumes, timeout=60 * MINUTES)
def fetch_weights() -> None:
    """The kit's installer: every pinned file at its commit, sha256-checked."""
    subprocess.run(
        ["bash", "run.sh", "install", "--weights", _HF_HOME], cwd=_KIT, check=True
    )
    weights_vol.commit()


@app.function(image=image, volumes=_volumes, gpu=GPU, timeout=90 * MINUTES)
def gpu_test() -> str:
    """tests/test_gpu_model_opt_esmfold2.py: exact vs off bitwise; fast engages."""
    env = dict(os.environ, SHEAF_GPU_KIT_TEST="1", MODEL_OPT_JIT_ROOT=_JIT)
    env["SHEAF_GPU_KIT_ESMFOLD2_MODEL"] = MODEL
    proc = subprocess.run(
        ["python", "-m", "pytest", "-v", "-rP", "tests/test_gpu_model_opt_esmfold2.py"],
        cwd=_SHEAF,
        env=env,
        capture_output=True,
        text=True,
    )
    jit_vol.commit()
    print(proc.stdout, proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU test failed (rc={proc.returncode})")
    return proc.stdout


@app.function(image=image, volumes=_volumes, gpu=GPU, timeout=120 * MINUTES)
def run_bench(extra_args: list[str]) -> dict[str, str]:
    out = "/tmp/esmfold2-bench"
    subprocess.run(
        [
            "python",
            "bench/model_opt/bench_esmfold2.py",
            "--model",
            MODEL,
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
    name = MODEL.split("/")[-1].lower()
    dest = _REPO / "bench" / "results" / f"{stamp}-{name}-kit-{GPU.lower()}"
    dest.mkdir(parents=True, exist_ok=True)
    for fname, text in files.items():
        (dest / fname).write_text(text)
    print(f"wrote {dest}")
    print(files.get("README.md", ""))
