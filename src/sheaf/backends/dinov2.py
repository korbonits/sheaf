"""DINOv2 backend for image embeddings via HuggingFace transformers.

Requires: pip install "sheaf-serve[vision]"
Models (HuggingFace Hub):
  "facebook/dinov2-small"  — 384-dim,  21M params
  "facebook/dinov2-base"   — 768-dim,  86M params  (default)
  "facebook/dinov2-large"  — 1024-dim, 307M params
  "facebook/dinov2-giant"  — 1536-dim, 1.1B params

DINOv2 is a vision-only self-supervised backbone — it has no text encoder.
Use OpenCLIPBackend for cross-modal (image + text) embeddings.

Pooling strategies:
  "cls"  — CLS token at position 0 of last_hidden_state (default).
            Best for retrieval, classification, kNN search.
  "mean" — Mean of patch tokens (positions 1:). Slightly smoother
            representation; useful for dense similarity tasks.

PIL.Image is stored as an instance attribute at load() time so the
dependency stays lazy and tests can inject a mock without PIL installed.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("dinov2")
class DINOv2Backend(ModelBackend):
    """ModelBackend for DINOv2 image embeddings (facebook/dinov2-*).

    Accepts base64-encoded images; rejects text inputs (image-only backbone).
    Returns fixed-size float vectors, L2-normalized by default.

    Args:
        model_name: HuggingFace model ID. Options:
            "facebook/dinov2-small"  (384-dim)
            "facebook/dinov2-base"   (768-dim, default)
            "facebook/dinov2-large"  (1024-dim)
            "facebook/dinov2-giant"  (1536-dim)
        device: "cpu", "cuda", or "mps"
        pooling: "cls" uses the CLS token (position 0); "mean" averages
            all patch tokens (positions 1:).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cpu",
        pooling: Literal["cls", "mean"] = "cls",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._pooling = pooling
        self._model: Any = None
        self._processor: Any = None
        self._Image: Any = None  # PIL.Image, injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.EMBEDDING

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from transformers import (  # ty: ignore[unresolved-import]
                AutoImageProcessor,
                AutoModel,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the DINOv2 backend. "
                "Install it with: pip install 'sheaf-serve[vision]'"
            ) from e
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, EmbeddingRequest):
            raise TypeError(f"Expected EmbeddingRequest, got {type(request)}")
        if request.texts is not None:
            raise ValueError(
                "DINOv2Backend only supports image embeddings. "
                "Provide images_b64, not texts."
            )
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        assert request.images_b64 is not None
        imgs = [
            self._Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            for b64 in request.images_b64
        ]

        inputs = self._processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        hidden = outputs.last_hidden_state  # (N, seq_len, dim)
        if self._pooling == "cls":
            embs = hidden[:, 0, :]  # CLS token
        else:
            embs = hidden[:, 1:, :].mean(dim=1)  # mean of patch tokens

        if request.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)

        embs_list: list[list[float]] = embs.cpu().float().tolist()

        return EmbeddingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=embs_list,
            dim=embs.shape[-1],
        )
