"""API contract for protein language models (ESMC, etc.).

This is distinct from :mod:`sheaf.api.molecular`, which covers per-sequence
pooled protein embeddings (ESM-3). ESMC and its successors return *per-token*
logits and optionally per-token hidden-state embeddings — a ragged tensor,
not a fixed-size vector per sequence.
"""

from __future__ import annotations

from typing import Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class ProteinLanguageRequest(BaseRequest):
    """Request contract for protein language model inference.

    A single request runs a batch of amino-acid sequences through one
    forward pass. The model returns per-token logits over the amino-acid
    vocabulary and (optionally) per-token hidden-state embeddings.

    Attributes:
        sequences: List of amino-acid sequences. Standard single-letter
            codes (ACDEFGHIKLMNPQRSTVWY plus ambiguity codes accepted by
            the ESM tokenizer).
        return_logits: If True (default), return per-token logits over
            the amino-acid vocabulary. Shape per sequence: ``(L_i, V)``.
        return_embeddings: If True, return per-token last-layer hidden
            states. Shape per sequence: ``(L_i, H)``. Defaults to False
            — these can be very large for long sequences against a
            6B-parameter model.
        output_hidden_states: If True, return per-token hidden states
            from *every* transformer layer (the upstream
            ``output_hidden_states=True`` flag on the HF model). Defaults
            to False. Implies ``return_embeddings=True``.
    """

    model_type: Literal[ModelType.PROTEIN_LANGUAGE] = ModelType.PROTEIN_LANGUAGE

    sequences: list[str]
    return_logits: bool = True
    return_embeddings: bool = False
    output_hidden_states: bool = False


class ProteinLanguageResponse(BaseResponse):
    """Response contract for protein language model inference.

    All ragged per-sequence tensors are returned as nested lists sliced
    back to each input sequence's tokenized length (``seq_lens[i]``) — i.e.
    padding is stripped before serialisation.

    Attributes:
        logits: Per-sequence per-token logits over the amino-acid
            vocabulary. ``logits[i]`` has shape ``(seq_lens[i], V)``.
            ``None`` when ``return_logits=False``.
        embeddings: Per-sequence per-token last-layer hidden states.
            ``embeddings[i]`` has shape ``(seq_lens[i], H)``. ``None``
            when ``return_embeddings=False`` and
            ``output_hidden_states=False``.
        hidden_states: Per-layer per-sequence per-token hidden states.
            ``hidden_states[layer][i]`` has shape ``(seq_lens[i], H)``.
            ``None`` unless ``output_hidden_states=True``.
        seq_lens: Tokenized length of each input sequence (includes any
            BOS/EOS special tokens the tokenizer adds — same convention
            as the upstream attention mask).
        vocab_size: Size of the amino-acid vocabulary (``V`` above).
        hidden_dim: Hidden-state dimensionality (``H`` above).
    """

    model_type: Literal[ModelType.PROTEIN_LANGUAGE] = ModelType.PROTEIN_LANGUAGE

    logits: list[list[list[float]]] | None = None
    embeddings: list[list[list[float]]] | None = None
    hidden_states: list[list[list[list[float]]]] | None = None
    seq_lens: list[int]
    vocab_size: int | None = None
    hidden_dim: int | None = None
