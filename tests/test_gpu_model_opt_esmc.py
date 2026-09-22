"""GPU equivalence test: ESM C kit ``exact`` must be bitwise identical to ``off``.

Gated on ``SHEAF_GPU_KIT_TEST=1``.  Needs the ESM C kit image (esm 3.4.0 @
43ccece2 + the kit's pinned stack + ``esmc_opt``), a CUDA GPU and the pinned
weights under ``HF_HOME`` — ``bench/model_opt/modal_esmc.py::gpu_test`` runs
it on a Modal H100.

``off`` and ``exact`` run in separate subprocesses (the kit patches the
interpreter process-wide), each over the fixed sequence set and batch
compositions in ``tests/gpu_kit/sequences.py``.  Every logits / embeddings
array is compared bit for bit (float32 viewed as uint32), not with a
tolerance: ``exact`` promises identical bytes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SHEAF_GPU_KIT_TEST"),
    reason="Set SHEAF_GPU_KIT_TEST=1 (ESM C kit image + GPU) to run",
)

_REPO = Path(__file__).resolve().parent.parent


def _dump(mode: str, out: Path) -> str:
    cmd = [sys.executable, "-m", "tests.gpu_kit.esmc_dump", "--mode", mode]
    cmd += ["--out", str(out)]
    model = os.environ.get("SHEAF_GPU_KIT_ESMC_MODEL")
    if model:
        cmd += ["--model", model]
    env = dict(os.environ)
    # Belt and braces: the kit's env hook must not engage in the off process.
    env.pop("ESMC_OPT", None)
    proc = subprocess.run(
        cmd, cwd=_REPO, env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"{mode} run failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout + proc.stderr


def test_exact_is_bitwise_identical_to_off(tmp_path: Path) -> None:
    off_log = _dump("off", tmp_path / "off.npz")
    exact_log = _dump("exact", tmp_path / "exact.npz")

    assert "[esmc-opt]" not in off_log, "the kit engaged under mode=off"
    assert "[esmc-opt] ACTIVE mode=exact" in exact_log
    assert "levers_fallback=none" in exact_log, exact_log

    off = np.load(tmp_path / "off.npz")
    exact = np.load(tmp_path / "exact.npz")
    assert sorted(off.files) == sorted(exact.files)
    mismatched = [
        k
        for k in off.files
        if off[k].shape != exact[k].shape
        or not np.array_equal(off[k].view(np.uint32), exact[k].view(np.uint32))
    ]
    assert not mismatched, f"{len(mismatched)} arrays differ: {mismatched[:5]}"
