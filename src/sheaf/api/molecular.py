"""API contract for molecular / protein language models (ESM-3, etc.)."""

from __future__ import annotations

from typing import Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class MolecularRequest(BaseRequest):
    """Request contract for protein sequence embedding.

    A single request embeds a batch of protein sequences.  Sequences should
    use standard single-letter amino acid codes (ACDEFGHIKLMNPQRSTVWY plus
    ambiguity codes accepted by ESM tokenizers).

    Args:
        sequences: List of amino acid sequences to embed.
        pooling: How to reduce the per-residue hidden states to a single
            vector per sequence.
            ``"mean"`` (default) — mean over residue positions (excludes
            BOS/EOS special tokens at positions 0 and -1).
            ``"cls"`` — BOS token at position 0 (analogous to [CLS] in BERT).
        normalize: If True (default), L2-normalize each embedding so that
            cosine similarity equals dot product.
    """

    model_type: Literal[ModelType.MOLECULAR] = ModelType.MOLECULAR

    sequences: list[str]
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True


class MolecularResponse(BaseResponse):
    """Response contract for protein sequence embedding."""

    model_type: Literal[ModelType.MOLECULAR] = ModelType.MOLECULAR

    # Embedding matrix — shape (N, dim) where N = len(sequences)
    embeddings: list[list[float]]

    # Embedding dimensionality (e.g. 1536 for esm3-sm-open-v1)
    dim: int
