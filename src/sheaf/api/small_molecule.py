"""API contract for small molecule / chemical foundation models (MolFormer, etc.)."""

from __future__ import annotations

from typing import Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class SmallMoleculeRequest(BaseRequest):
    """Request contract for small molecule embedding.

    A single request embeds a batch of chemical compounds given as SMILES
    strings. SMILES (Simplified Molecular-Input Line-Entry System) is the
    standard text representation of molecular structure.

    Args:
        smiles: List of SMILES strings to embed. Each string represents one
            molecule (e.g. "CC(=O)OC1=CC=CC=C1C(=O)O" for aspirin).
        pooling: How to reduce per-token hidden states to a fixed-size vector.
            ``"mean"`` (default) — attention-masked mean over all tokens,
            excluding padding. Best for molecular property prediction.
            ``"cls"`` — CLS token at position 0. Useful for models with a
            dedicated classification token.
        normalize: If True, L2-normalize each embedding (cosine similarity ==
            dot product). Defaults to ``False`` — raw embeddings are more
            natural for regression tasks such as property prediction.
    """

    model_type: Literal[ModelType.SMALL_MOLECULE] = ModelType.SMALL_MOLECULE

    smiles: list[str]
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = False


class SmallMoleculeResponse(BaseResponse):
    """Response contract for small molecule embedding."""

    model_type: Literal[ModelType.SMALL_MOLECULE] = ModelType.SMALL_MOLECULE

    # Embedding matrix — shape (N, dim) where N = len(smiles)
    embeddings: list[list[float]]

    # Embedding dimensionality (768 for MoLFormer-XL)
    dim: int
