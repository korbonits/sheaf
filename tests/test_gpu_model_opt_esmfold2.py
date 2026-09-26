"""GPU checks for the ESMFold2 kit: ``exact`` bitwise == ``off``; ``fast`` engages.

Gated on ``SHEAF_GPU_KIT_TEST=1``.  Needs the ESMFold2 kit image (esm 3.3.0 @
26b0bc2b + the kit's pinned stack + ``esmfold2_opt``), a CUDA GPU and the
pinned weights under ``HF_HOME`` — ``bench/model_opt/modal_esmfold2.py::gpu_test``
runs it on a Modal H100.

Each mode runs in its own subprocess (the kit patches the interpreter
process-wide) over the fixed inputs and seeds of ``tests/gpu_kit/esmfold2_dump.py``,
under the kit's deterministic recipe.

- ``exact``: the structure text and every pLDDT / PAE / pTM array must equal
  ``off``'s bit for bit (``off`` = the fused backend, the configuration
  ``exact`` reproduces).  No tolerance.
- ``fast``: documented as "within stock's seed-to-seed variation", not bitwise.
  The test requires full activation (the backend refuses at load when any lever
  is partial) and finite outputs of the same shapes, and prints how far ``fast``
  is from ``off`` next to how far ``off``'s two seeds are from each other.  It
  asserts no numeric threshold: the kit states none to test against.
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
    reason="Set SHEAF_GPU_KIT_TEST=1 (ESMFold2 kit image + GPU) to run",
)

_REPO = Path(__file__).resolve().parent.parent


def _dump(mode: str, out: Path) -> str:
    cmd = [sys.executable, "-m", "tests.gpu_kit.esmfold2_dump", "--mode", mode]
    cmd += ["--out", str(out)]
    model = os.environ.get("SHEAF_GPU_KIT_ESMFOLD2_MODEL")
    if model:
        cmd += ["--model", model]
    env = dict(os.environ)
    env.pop("ESMFOLD2_OPT", None)  # the explicit enable() path only
    proc = subprocess.run(
        cmd, cwd=_REPO, env=env, capture_output=True, text=True, check=False
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{mode} run failed:\n{log}"
    return log


@pytest.fixture(scope="module")
def off(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, dict]:
    path = tmp_path_factory.mktemp("esmfold2") / "off.npz"
    log = _dump("off", path)
    return log, dict(np.load(path))


def test_exact_is_bitwise_identical_to_off(
    off: tuple[str, dict], tmp_path: Path
) -> None:
    off_log, off_arrays = off
    exact_log = _dump("exact", tmp_path / "exact.npz")
    exact_arrays = dict(np.load(tmp_path / "exact.npz"))
    print(f"--- off ---\n{off_log}\n--- exact ---\n{exact_log}")

    assert "[esmfold2-opt]" not in off_log, "the kit engaged under mode=off"
    assert "[esmfold2-opt] ACTIVE mode=exact" in exact_log
    assert sorted(off_arrays) == sorted(exact_arrays)
    mismatched = [
        k
        for k in off_arrays
        if off_arrays[k].shape != exact_arrays[k].shape
        or off_arrays[k].tobytes() != exact_arrays[k].tobytes()
    ]
    assert not mismatched, f"{len(mismatched)} arrays differ: {mismatched[:5]}"


def test_fast_engages_and_reports_distance_from_off(
    off: tuple[str, dict], tmp_path: Path
) -> None:
    _, off_arrays = off
    fast_log = _dump("fast", tmp_path / "fast.npz")
    fast_arrays = dict(np.load(tmp_path / "fast.npz"))
    print(f"--- fast ---\n{fast_log}")

    assert "[esmfold2-opt] ACTIVE mode=fast" in fast_log
    assert sorted(off_arrays) == sorted(fast_arrays)
    for k, a in fast_arrays.items():
        assert a.shape == off_arrays[k].shape, k
        if not k.endswith("/structure"):
            assert np.isfinite(a).all() or k.endswith("/ptm"), k

    def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))

    names = sorted({k.split("/")[0] for k in off_arrays})
    print(
        "input        | pLDDT |fast-off| s0 | pLDDT |off s0-s1| "
        "| pTM |fast-off| s0 | pTM |off s0-s1|"
    )
    for n in names:
        p_fast = mean_abs(fast_arrays[f"{n}/s0/plddt"], off_arrays[f"{n}/s0/plddt"])
        p_seed = mean_abs(off_arrays[f"{n}/s0/plddt"], off_arrays[f"{n}/s1/plddt"])
        t_fast = mean_abs(fast_arrays[f"{n}/s0/ptm"], off_arrays[f"{n}/s0/ptm"])
        t_seed = mean_abs(off_arrays[f"{n}/s0/ptm"], off_arrays[f"{n}/s1/ptm"])
        print(f"{n:<12} | {p_fast:.4f} | {p_seed:.4f} | {t_fast:.4f} | {t_seed:.4f}")
