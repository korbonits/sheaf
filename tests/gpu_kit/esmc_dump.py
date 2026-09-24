"""Run ESMCBackend under one model_opt mode and dump its outputs to .npz.

Kits patch the interpreter process-wide, so ``off`` and ``exact`` must be
produced by separate processes — this script is that process:

    python -m tests.gpu_kit.esmc_dump --mode off   --out off.npz
    python -m tests.gpu_kit.esmc_dump --mode exact --out exact.npz

Each ``BATCHES`` entry is one ``ProteinLanguageRequest``; its logits and
last-layer embeddings (float32, sliced to each sequence's length, exactly
what the response carries) are stored per sequence.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sheaf.api.protein_language import ProteinLanguageRequest
from sheaf.backends.esmc import ESMCBackend
from sheaf.model_opt import ModelOptConfig, apply_model_opt
from tests.gpu_kit.sequences import BATCHES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["off", "exact"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Biohub/ESMC-6B")
    ap.add_argument("--jit-root", default=None)
    args = ap.parse_args()

    backend = ESMCBackend(model_name=args.model, device="cuda")
    apply_model_opt(
        backend, ModelOptConfig(mode=args.mode, jit_root=args.jit_root), "gpu-test"
    )
    backend.load()

    arrays: dict[str, np.ndarray] = {}
    for name, seqs in BATCHES.items():
        resp = backend.predict(
            ProteinLanguageRequest(
                model_name="esmc",
                sequences=seqs,
                return_logits=True,
                return_embeddings=True,
            )
        )
        for i in range(len(seqs)):
            arrays[f"{name}/{i}/logits"] = np.asarray(resp.logits[i], np.float32)
            arrays[f"{name}/{i}/embeddings"] = np.asarray(
                resp.embeddings[i], np.float32
            )
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
