"""API contract for DNA/genomic foundation models (Nucleotide Transformer, etc.)."""

from __future__ import annotations

from typing import Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class GenomicRequest(BaseRequest):
    """Request contract for DNA/RNA sequence embedding.

    A single request embeds a batch of nucleotide sequences. Sequences should
    use standard nucleotide codes (A, C, G, T for DNA; A, C, G, U for RNA;
    N for unknown bases). All are accepted by Nucleotide Transformer tokenizers.

    Args:
        sequences: List of nucleotide sequences to embed.
        pooling: How to reduce per-token hidden states to a single vector.
            ``"mean"`` (default) — mean of non-special tokens (excludes CLS
            at position 0 and EOS/SEP at position -1).
            ``"cls"`` — CLS token at position 0 (analogous to [CLS] in BERT).
        normalize: If True (default), L2-normalize each embedding so that
            cosine similarity equals dot product.
    """

    model_type: Literal[ModelType.GENOMIC] = ModelType.GENOMIC

    sequences: list[str]
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True


class GenomicResponse(BaseResponse):
    """Response contract for DNA/RNA sequence embedding."""

    model_type: Literal[ModelType.GENOMIC] = ModelType.GENOMIC

    # Embedding matrix — shape (N, dim) where N = len(sequences)
    embeddings: list[list[float]]

    # Embedding dimensionality (e.g. 512 for NT-v2-100M, 1280 for NT-v2-500M)
    dim: int
