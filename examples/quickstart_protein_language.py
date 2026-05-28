"""ESMC protein language model quickstart.

Requirements:
    pip install "sheaf-serve[protein]"
    pip install esm@git+https://github.com/Biohub/esm.git@c94ed8d
    # Python 3.12+ required. Conflicts with the [molecular] extra (ESM-3).

Usage:
    python examples/quickstart_protein_language.py

Demonstrates:
  - Per-token logits over the amino-acid vocabulary
  - Per-token last-layer hidden-state embeddings
  - Per-sequence mean-pooled embeddings (call-site reduction)

ESMC 6B effectively requires a GPU with bf16 support; the default
``device="cuda"`` below will fall back to CPU only on a small dummy
checkpoint. For a real run, set ``CUDA_VISIBLE_DEVICES`` and leave the
device as is.
"""

from __future__ import annotations

import math

from sheaf.api.protein_language import ProteinLanguageRequest
from sheaf.backends.esmc import ESMCBackend

# Three short protein sequences — variable length, exercises the padding /
# attention-mask slicing path.
SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",  # 53 aa
    "ACDEFGHIKLMNPQRSTVWY",  # 20 aa — one of each standard residue
    "MVLSPADKTNVKAAW",  # 15 aa — N-terminus of human hemoglobin alpha
]

print("--- ESMC (Biohub/ESMC-6B) ---")
print("Loading model (downloads ~12 GB of weights on first run)...")

esmc = ESMCBackend(model_name="Biohub/ESMC-6B", device="cuda")
esmc.load()
print("Model loaded.")

# Per-token logits over the amino-acid vocab — shape (L_i, V) per sequence.
req = ProteinLanguageRequest(
    model_name="esmc",
    sequences=SEQUENCES,
    return_logits=True,
    return_embeddings=True,
)
resp = esmc.predict(req)

print(f"\nVocab size  : {resp.vocab_size}")
print(f"Hidden dim  : {resp.hidden_dim}")
print(f"Seq lengths : {resp.seq_lens}  (includes BOS/EOS special tokens)")

assert resp.logits is not None
assert resp.embeddings is not None
print("\nPer-token logits shape per sequence:")
for i, (seq, li) in enumerate(zip(SEQUENCES, resp.logits)):
    print(f"  seq[{i}] len={len(seq):3d}  logits=({len(li)}, {len(li[0])})")

print("\nPer-token embeddings shape per sequence:")
for i, (seq, ei) in enumerate(zip(SEQUENCES, resp.embeddings)):
    print(f"  seq[{i}] len={len(seq):3d}  embeddings=({len(ei)}, {len(ei[0])})")

# Mean-pool per sequence for a fixed-size representation (call-site reduction —
# Sheaf returns the raw ragged tensor; pooling is a caller policy choice).
print("\nMean-pooled per-sequence embeddings (first 4 dims):")
for i, ei in enumerate(resp.embeddings):
    n_tokens = len(ei)
    pooled = [sum(tok[d] for tok in ei) / n_tokens for d in range(len(ei[0]))]
    print(f"  seq[{i}]  {[f'{x:+.3f}' for x in pooled[:4]]}")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


print("\nPairwise cosine similarity of mean-pooled embeddings:")
pooled = []
for ei in resp.embeddings:
    n_tokens = len(ei)
    pooled.append([sum(tok[d] for tok in ei) / n_tokens for d in range(len(ei[0]))])
for i in range(len(SEQUENCES)):
    for j in range(i + 1, len(SEQUENCES)):
        print(f"  seq[{i}] vs seq[{j}]  {cosine(pooled[i], pooled[j]):+.4f}")
