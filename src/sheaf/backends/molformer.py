"""MolFormer backend for small molecule embeddings via HuggingFace transformers.

Requires: pip install "sheaf-serve[small-molecule]"
Library:  transformers (trust_remote_code=True for MolFormer custom classes)

MolFormer-XL is a linear-attention transformer pretrained on ~1.1 billion
SMILES strings (PubChem + ZINC). It produces 768-dim embeddings suitable for
molecular property prediction, virtual screening, and similarity search.

Supported models (HuggingFace Hub):
  "ibm/MoLFormer-XL-both-10pct"   — pretrained on 10% data sample (default)
  "ibm/MoLFormer-XL-both-100pct"  — pretrained on full dataset

Tokenization: character-level SMILES tokenizer with CLS/EOS/PAD tokens.
A batch of molecules is tokenized together with padding to the longest sequence.

Pooling strategies:
  "mean" — attention-masked mean over all token positions (default). Padding
            tokens are excluded via the attention mask. Best for property
            prediction and regression tasks.
  "cls"  — CLS token at position 0. Analogous to [CLS] in BERT.

AutoTokenizer and AutoModel are stored at load() time so tests can inject
mocks without transformers installed in the test environment.
"""

from __future__ import annotations

from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.small_molecule import SmallMoleculeRequest, SmallMoleculeResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_DEFAULT_MODEL = "ibm/MoLFormer-XL-both-10pct"


@register_backend("molformer")
class MolFormerBackend(ModelBackend):
    """ModelBackend for small molecule embeddings via MolFormer-XL.

    A batch of SMILES strings is tokenized together (with padding) and passed
    through MolFormer in a single forward pass. Per-token hidden states are
    pooled to fixed-size vectors using the attention mask.

    Args:
        model_name: HuggingFace model ID (ibm/MoLFormer-XL-*).
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
        return ModelType.SMALL_MOLECULE

    def load(self) -> None:
        try:
            from transformers import (  # ty: ignore[unresolved-import]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the MolFormer backend. "
                "Install it with: pip install 'sheaf-serve[small-molecule]'"
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, SmallMoleculeRequest):
            raise TypeError(f"Expected SmallMoleculeRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: SmallMoleculeRequest) -> SmallMoleculeResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        # Tokenize the full batch at once — padding to longest sequence.
        inputs = self._tokenizer(
            request.smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self._model(**inputs)

        hidden = output.last_hidden_state  # (N, L, D)
        attention_mask = inputs["attention_mask"]  # (N, L)

        if request.pooling == "cls":
            embs = hidden[:, 0, :]  # (N, D)
        else:
            # Attention-masked mean: exclude padding tokens.
            mask_exp = attention_mask.unsqueeze(-1).float()  # (N, L, 1)
            sum_h = (hidden * mask_exp).sum(dim=1)  # (N, D)
            count = mask_exp.sum(dim=1)  # (N, 1)
            embs = sum_h / count  # (N, D)

        if request.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)

        embs_list: list[list[float]] = embs.cpu().float().tolist()
        dim = len(embs_list[0]) if embs_list else 0

        return SmallMoleculeResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=embs_list,
            dim=dim,
        )
