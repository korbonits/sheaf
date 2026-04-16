"""Nucleotide Transformer backend for DNA/RNA sequence embeddings.

Requires: pip install "sheaf-serve[genomics]"
Library:  transformers (HuggingFace)

Supported models (HuggingFace Hub — InstaDeepAI/EMBL-EBI):
  "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"  — 100M (default)
  "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"  — 500M
  "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"     — 2.5B
  "InstaDeepAI/nucleotide-transformer-2.5b-1000g"             — 2.5B, 1000 Genomes

Tokenization: 6-mer overlapping k-mers. A sequence of length L produces
approximately L/6 tokens plus CLS and EOS special tokens.

Pooling strategies:
  "mean" — mean of non-special token hidden states (positions 1:-1, excluding
            CLS at position 0 and EOS/SEP at position -1). Best for
            sequence-level similarity and retrieval.
  "cls"  — CLS token at position 0. Analogous to [CLS] in BERT.

AutoTokenizer and AutoModel are stored as instance attributes at load() time
so tests can inject mocks without transformers installed in the test env.
"""

from __future__ import annotations

from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.genomic import GenomicRequest, GenomicResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"


@register_backend("nucleotide_transformer")
class NucleotideTransformerBackend(ModelBackend):
    """ModelBackend for DNA/RNA sequence embeddings via Nucleotide Transformer.

    Each sequence in the request is tokenized and passed through the model
    independently. Per-token hidden states are pooled to a fixed-size vector.
    Embeddings are L2-normalized by default.

    Args:
        model_name: HuggingFace model ID (InstaDeepAI/nucleotide-transformer-*).
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None  # stored at load() for test injectability

    @property
    def model_type(self) -> str:
        return ModelType.GENOMIC

    def load(self) -> None:
        try:
            from transformers import (  # ty: ignore[unresolved-import]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the NucleotideTransformer backend. "
                "Install it with: pip install 'sheaf-serve[genomics]'"
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, GenomicRequest):
            raise TypeError(f"Expected GenomicRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: GenomicRequest) -> GenomicResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        all_embs: list[list[float]] = []
        for seq in request.sequences:
            inputs = self._tokenizer(
                seq,
                return_tensors="pt",
                truncation=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model(**inputs)

            hidden = output.last_hidden_state  # (1, L, D)

            if request.pooling == "cls":
                emb = hidden[:, 0, :]  # (1, D) — CLS token
            else:
                emb = hidden[:, 1:-1, :].mean(dim=1)  # (1, D) — excl. CLS & EOS

            if request.normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True)

            all_embs.append(emb.cpu().float().tolist()[0])

        dim = len(all_embs[0]) if all_embs else 0
        return GenomicResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=all_embs,
            dim=dim,
        )
