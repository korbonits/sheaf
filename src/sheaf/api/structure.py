"""API contract for protein structure prediction (ESMFold2, etc.).

Structure prediction is a new top-level model category in Sheaf — outputs
are 3D atomic coordinates (PDB / mmCIF strings) plus confidence side-channels
(pLDDT, pTM, ipTM, PAE), not embeddings or classifications. See
``docs/adr/0001-esmc-esmfold2-integration.md`` for the rationale.

Inference-time scaling parameters (``num_loops``, ``num_sampling_steps``,
``num_samples``) are exposed as first-class request fields, not hidden
kwargs — this is the headline capability of ESMFold2's looped-transformer
architecture and a key Sheaf differentiator vs. plain HF serving.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class ChainInput(BaseModel):
    """A single chain in a structure-prediction request.

    Multi-chain complexes are expressed by passing multiple ``ChainInput``s
    in ``StructureRequest.chains``. Ligand / DNA / modified-residue inputs
    are out of scope for v0.11 — the upstream ``StructurePredictionInput``
    supports them but they require additional Pydantic shape work; this
    contract is protein-chains-only.

    Attributes:
        chain_id: Single-letter (or short) chain identifier — used as the
            chain ID in the output PDB / mmCIF. Must be unique within a
            request.
        sequence: Amino-acid sequence using standard single-letter codes.
    """

    chain_id: str
    sequence: str


class StructureRequest(BaseRequest):
    """Request contract for protein structure prediction.

    A single request predicts one structure (possibly multi-chain). The
    inference-time scaling parameters control the depth of the
    looped-transformer recurrence (``num_loops``), the number of diffusion
    sampling steps (``num_sampling_steps``), and how many independent
    samples to draw (``num_samples``); when ``num_samples > 1`` the
    response includes per-sample ranking scores and returns the
    highest-scoring structure as the primary output.

    Attributes:
        chains: One or more protein chains to fold together. Single-chain
            structure prediction uses a one-element list.
        msa: Optional multiple-sequence alignment input, one list of
            homolog sequences per chain (parallel to ``chains``). ESMFold2
            runs in single-sequence mode when ``msa`` is None (the headline
            speedup path); supplying an MSA improves accuracy on
            challenging targets at the cost of throughput.
        num_loops: Depth of the looped-transformer recurrence. Default 3.
        num_sampling_steps: Number of diffusion sampling steps per sample.
            Default 50.
        num_samples: Number of independent structures to sample. Default
            1; values > 1 enable self-confidence-based ranking.
        seed: Random seed for the diffusion sampler. Default 0.
        output_format: ``"mmcif"`` (default) or ``"pdb"``.
    """

    model_type: Literal[ModelType.STRUCTURE] = ModelType.STRUCTURE

    chains: list[ChainInput] = Field(min_length=1)
    msa: list[list[str]] | None = None
    num_loops: int = 3
    num_sampling_steps: int = 50
    num_samples: int = 1
    seed: int = 0
    output_format: Literal["mmcif", "pdb"] = "mmcif"


class StructureResponse(BaseResponse):
    """Response contract for protein structure prediction.

    Attributes:
        structure: The predicted structure encoded as PDB or mmCIF text
            (per ``StructureRequest.output_format``). When
            ``num_samples > 1`` this is the highest-confidence sample.
        structure_format: ``"mmcif"`` or ``"pdb"``, matching the request.
        plddt: Per-residue predicted-Local-Distance-Difference-Test score.
            ESMFold2 reports pLDDT on **[0, 1]** (fractional), not the
            conventional AlphaFold / ESMFold-v1 [0, 100] scale — multiply by
            100 if you need the conventional values. Length = total residues
            across all chains.
        ptm: Predicted-TM score for the structure as a whole. ``None`` if
            the model did not produce one.
        iptm: Interface-pTM (for multi-chain complexes). ``None`` for
            single-chain inputs or when the model did not produce one.
        pae: Predicted Aligned Error matrix, shape ``(N, N)`` where N is
            total residues. ``None`` if the model did not produce one
            (e.g. some fast variants skip PAE to save compute).
        sample_scores: When ``num_samples > 1``, the self-confidence
            score for each sample in the order produced. The returned
            ``structure`` corresponds to ``argmax(sample_scores)``.
            ``None`` when ``num_samples == 1``.
    """

    model_type: Literal[ModelType.STRUCTURE] = ModelType.STRUCTURE

    structure: str
    structure_format: Literal["mmcif", "pdb"]
    plddt: list[float]
    ptm: float | None = None
    iptm: float | None = None
    pae: list[list[float]] | None = None
    sample_scores: list[float] | None = None
