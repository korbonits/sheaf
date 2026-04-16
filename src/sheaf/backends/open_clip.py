"""OpenCLIP backend for image and text embeddings.

Requires: pip install "sheaf-serve[vision]"
Library: open-clip-torch (https://github.com/mlfoundations/open_clip)

Supported architectures (examples):
  "ViT-B-32"  / pretrained="openai"        — OpenAI CLIP ViT-B/32
  "ViT-L-14"  / pretrained="openai"        — OpenAI CLIP ViT-L/14
  "ViT-H-14"  / pretrained="laion2b_s32b_b79k"  — OpenCLIP ViT-H/14
  "ViT-B-16"  / pretrained="laion400m_e32" — OpenCLIP ViT-B/16
  "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" — domain FMs

Full list: open_clip.list_pretrained()

PIL.Image is stored as an instance attribute at load() time so the
dependency stays lazy and tests can inject a mock without PIL installed.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("open-clip")
class OpenCLIPBackend(ModelBackend):
    """ModelBackend for OpenCLIP (image and text embeddings).

    Encodes batches of text strings or base64-encoded images into
    fixed-size float vectors. Embeddings are L2-normalized by default,
    making cosine similarity equivalent to dot product.

    Args:
        model_name: Architecture identifier, e.g. "ViT-B-32", "ViT-L-14".
            Accepts HuggingFace Hub model IDs prefixed with "hf-hub:".
        pretrained: Pretrained weight tag, e.g. "openai", "laion2b_s32b_b79k".
            Use open_clip.list_pretrained() to see available combinations.
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._Image: Any = None  # PIL.Image, injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.EMBEDDING

    def load(self) -> None:
        try:
            import open_clip  # ty: ignore[unresolved-import]
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "open-clip-torch is required for the OpenCLIP backend. "
                "Install it with: pip install 'sheaf-serve[vision]'"
            ) from e
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name,
            pretrained=self._pretrained,
            device=self._device,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, EmbeddingRequest):
            raise TypeError(f"Expected EmbeddingRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        if request.texts is not None:
            tokens = self._tokenizer(request.texts)
            if hasattr(tokens, "to"):
                tokens = tokens.to(self._device)
            with torch.no_grad():
                embs = self._model.encode_text(tokens)
        else:
            assert request.images_b64 is not None
            imgs = [
                self._Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                for b64 in request.images_b64
            ]
            tensors = torch.stack([self._preprocess(img) for img in imgs]).to(
                self._device
            )
            with torch.no_grad():
                embs = self._model.encode_image(tensors)

        if request.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)

        embs_list: list[list[float]] = embs.cpu().float().tolist()

        return EmbeddingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=embs_list,
            dim=embs.shape[-1],
        )
