"""Run ESMFold2Backend under one model_opt mode and dump its outputs to .npz.

Kits patch the interpreter process-wide, so each mode is produced by its own
process — this script is that process:

    python -m tests.gpu_kit.esmfold2_dump --mode off   --out off.npz
    python -m tests.gpu_kit.esmfold2_dump --mode exact --out exact.npz

``exact`` equals ``off`` (the fused backend) bit for bit under the kit's
deterministic recipe (``esmfold2_opt/det.py`` level 1): the cuBLAS workspace
and the scatter switch set before torch is imported, deterministic
algorithms, and a seed per fold.  The recipe's fold keywords are inert here
(``lm_dropout=0`` is a no-op on the pinned build; ``msa_column_mask_rate``
only reads MSAs, which these inputs don't have), so every request goes
through ``backend.predict`` unchanged.  The recipe is stated here rather than
imported, so the ``off`` process imports nothing of the kit.

Per (input, seed) the structure text, pLDDT, PAE and pTM are stored exactly as
the response carries them.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("ESMFOLD2_DETERMINISTIC_SCATTER", "1")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

from sheaf.api.structure import ChainInput, StructureRequest  # noqa: E402
from sheaf.backends.esmfold2 import ESMFold2Backend  # noqa: E402
from sheaf.model_opt import ModelOptConfig, apply_model_opt  # noqa: E402
from tests.gpu_kit.sequences import (  # noqa: E402
    GFP,
    INSULIN_B,
    LYSOZYME,
    TRP_CAGE,
    UBIQUITIN,
)

INSULIN_A = "GIVEQCCTSICSLYQLENYCN"  # P01308 insulin A chain

INPUTS: dict[str, list[tuple[str, str]]] = {
    "ubiquitin": [("A", UBIQUITIN)],
    "trp_cage": [("A", TRP_CAGE)],
    "lysozyme": [("A", LYSOZYME)],
    "gfp": [("A", GFP)],
    "insulin_ab": [("A", INSULIN_A), ("B", INSULIN_B)],
}
SEEDS = (0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["off", "exact", "fast"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="biohub/ESMFold2")
    ap.add_argument("--jit-root", default=os.environ.get("MODEL_OPT_JIT_ROOT"))
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    import transformers.models.esmfold2.modeling_esmfold2_common as common  # ty: ignore[unresolved-import]

    if hasattr(common, "set_deterministic_scatter"):
        common.set_deterministic_scatter(True)

    backend = ESMFold2Backend(model_name=args.model, device="cuda")
    apply_model_opt(
        backend, ModelOptConfig(mode=args.mode, jit_root=args.jit_root), "gpu-test"
    )
    backend.load()

    arrays: dict[str, np.ndarray] = {}
    for name, chains in INPUTS.items():
        for seed in SEEDS:
            resp = backend.predict(
                StructureRequest(
                    model_name="esmfold2",
                    chains=[ChainInput(chain_id=c, sequence=s) for c, s in chains],
                    seed=seed,
                )
            )
            key = f"{name}/s{seed}"
            text = resp.structure.encode()
            arrays[f"{key}/structure"] = np.frombuffer(text, np.uint8)
            arrays[f"{key}/plddt"] = np.asarray(resp.plddt, np.float32)
            arrays[f"{key}/ptm"] = np.asarray(
                [np.nan if resp.ptm is None else resp.ptm], np.float64
            )
            if resp.pae is not None:
                arrays[f"{key}/pae"] = np.asarray(resp.pae, np.float32)
    np.savez(args.out, **arrays)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "arrays": len(arrays),
                "kit_lines": backend.model_opt_lines,
            }
        )
    )


if __name__ == "__main__":
    main()
