"""ESMFold2 protein structure prediction quickstart.

Requirements:
    pip install "sheaf-serve[protein]"
    pip install esm@git+https://github.com/Biohub/esm.git@c94ed8d
    # Python 3.12+ required. Conflicts with the [molecular] extra (ESM-3).

Usage:
    python examples/quickstart_structure.py

Demonstrates:
  - Single-chain structure prediction → PDB output
  - Multi-chain complex prediction → mmCIF + ipTM
  - Inference-time scaling via num_loops / num_sampling_steps / num_samples
  - Self-confidence-based ranking when num_samples > 1

ESMFold2 wraps a 6B-parameter ESMC backbone with a diffusion head; a GPU
with bf16 support is effectively required for real inference latencies.
"""

from __future__ import annotations

from pathlib import Path

from sheaf.api.structure import ChainInput, StructureRequest
from sheaf.backends.esmfold2 import ESMFold2Backend

# Single-chain target — N-terminus of bacteriophage T4 lysozyme.
SINGLE_CHAIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"

# Two-chain mini-complex for ipTM demonstration. (Toy sequences; in practice
# you would supply real interacting partners.)
CHAIN_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
CHAIN_B = "ACDEFGHIKLMNPQRSTVWY"

print("--- ESMFold2 (biohub/ESMFold2) ---")
print("Loading model (downloads several GB of weights on first run)...")

fold = ESMFold2Backend(model_name="biohub/ESMFold2", device="cuda")
fold.load()
print("Model loaded.")

# ---------------------------------------------------------------------------
# Single-chain fold → PDB
# ---------------------------------------------------------------------------

print("\n--- Single-chain fold, num_loops=3, num_sampling_steps=50 ---")
req = StructureRequest(
    model_name="esmfold2",
    chains=[ChainInput(chain_id="A", sequence=SINGLE_CHAIN)],
    num_loops=3,
    num_sampling_steps=50,
    output_format="pdb",
)
resp = fold.predict(req)

n_res = len(resp.plddt)
mean_plddt = sum(resp.plddt) / n_res if n_res else 0.0
print(f"  residues       : {n_res}")
print(f"  mean pLDDT     : {mean_plddt:.2f}")
print(f"  pTM            : {resp.ptm}")
print(f"  format         : {resp.structure_format}")
print(f"  structure size : {len(resp.structure)} chars")

out_pdb = Path("examples") / "esmfold2_single_chain.pdb"
out_pdb.write_text(resp.structure)
print(f"  wrote          : {out_pdb}")

# ---------------------------------------------------------------------------
# Multi-chain complex → mmCIF with ipTM
# ---------------------------------------------------------------------------

print("\n--- Two-chain complex, mmCIF output ---")
req_complex = StructureRequest(
    model_name="esmfold2",
    chains=[
        ChainInput(chain_id="A", sequence=CHAIN_A),
        ChainInput(chain_id="B", sequence=CHAIN_B),
    ],
    num_loops=3,
    num_sampling_steps=50,
    output_format="mmcif",
)
resp_complex = fold.predict(req_complex)

print(f"  total residues : {len(resp_complex.plddt)}")
print(f"  pTM            : {resp_complex.ptm}")
print(f"  ipTM           : {resp_complex.iptm}")

out_cif = Path("examples") / "esmfold2_complex.cif"
out_cif.write_text(resp_complex.structure)
print(f"  wrote          : {out_cif}")

# ---------------------------------------------------------------------------
# Inference-time scaling — multiple samples, ranked by self-confidence
# ---------------------------------------------------------------------------

print("\n--- Inference-time scaling: num_samples=4, ranked by self-confidence ---")
req_ranked = StructureRequest(
    model_name="esmfold2",
    chains=[ChainInput(chain_id="A", sequence=SINGLE_CHAIN)],
    num_loops=5,
    num_sampling_steps=100,
    num_samples=4,
    seed=42,
    output_format="mmcif",
)
resp_ranked = fold.predict(req_ranked)

assert resp_ranked.sample_scores is not None
print(f"  sample scores  : {[f'{s:.4f}' for s in resp_ranked.sample_scores]}")
print(
    f"  argmax sample  : "
    f"{resp_ranked.sample_scores.index(max(resp_ranked.sample_scores))}"
)
print("  → returned structure is the highest-confidence sample of the 4.")
