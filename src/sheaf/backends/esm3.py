"""ESM-3 backend for protein sequence embeddings via EvolutionaryScale's esm library.

Requires: pip install "sheaf-serve[molecular]"   (Python 3.12+)
Library:  esm (https://github.com/evolutionaryscale/esm)

Supported models (HuggingFace Hub — requires huggingface-cli login and
acceptance of the EvolutionaryScale license):
  "esm3-sm-open-v1"  — 1.4 B params, 1536-dim, open weights (default)

Pooling strategies:
  "mean" — mean of residue token hidden states (positions 1:-1, excluding
            BOS and EOS special tokens).  Best for sequence-level similarity.
  "cls"  — BOS token at position 0.  Analogous to [CLS] in BERT.

ESMProtein is stored as an instance attribute at load() time so the
dependency stays lazy and tests can inject a mock without esm installed.
"""

from __future__ import annotations

from typing import Any, Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.molecular import MolecularRequest, MolecularResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("esm3")
class ESM3Backend(ModelBackend):
    """ModelBackend for ESM-3 protein sequence embeddings.

    Each sequence in the request is encoded independently through the same
    model and the per-residue hidden states are pooled to a fixed-size vector.
    Embeddings are L2-normalized by default.

    Requires Python 3.12+ and a HuggingFace Hub login with the
    EvolutionaryScale license accepted.

    Args:
        model_name: HuggingFace / EvolutionaryScale model ID.
            Currently only "esm3-sm-open-v1" has open weights.
        device: "cpu", "cuda", or "mps"
        pooling: "mean" (default) or "cls"
    """

    def __init__(
        self,
        model_name: str = "esm3-sm-open-v1",
        device: str = "cpu",
        pooling: Literal["mean", "cls"] = "mean",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pooling = pooling
        self._model: Any = None
        self._ESMProtein: Any = None  # esm.sdk.api.ESMProtein, injected at load()

    @property
    def model_type(self) -> str:
        return ModelType.MOLECULAR

    def load(self) -> None:
        try:
            from esm.models.esm3 import ESM3  # ty: ignore[unresolved-import]
            from esm.sdk.api import ESMProtein  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "esm is required for the ESM3 backend. "
                "Install it with: pip install 'sheaf-serve[molecular]' "
                "(Python 3.12+ required)"
            ) from e
        self._model = ESM3.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._ESMProtein = ESMProtein

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, MolecularRequest):
            raise TypeError(f"Expected MolecularRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: MolecularRequest) -> MolecularResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        all_embs: list[list[float]] = []
        for seq in request.sequences:
            protein = self._ESMProtein(sequence=seq)
            protein_tensor = self._model.encode(protein)
            # sequence: (L,) int64 — L = len(seq) + 2 for BOS/EOS
            seq_tokens = protein_tensor.sequence.unsqueeze(0)  # (1, L)

            with torch.no_grad():
                output = self._model.forward(sequence_tokens=seq_tokens)

            hidden = output.embeddings  # (1, L, hidden_dim)
            if self._pooling == "cls":
                emb = hidden[:, 0, :]  # (1, hidden_dim)
            else:
                emb = hidden[:, 1:-1, :].mean(dim=1)  # (1, hidden_dim)

            if request.normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True)

            all_embs.append(emb.cpu().float().tolist()[0])

        dim = len(all_embs[0]) if all_embs else 0
        return MolecularResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=all_embs,
            dim=dim,
        )
