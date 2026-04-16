"""VideoMAE / TimeSformer backend for video understanding via HuggingFace transformers.

Supports any model compatible with AutoModel (embeddings) or
AutoModelForVideoClassification (classification) — VideoMAE, TimeSformer,
VideoMAE2, etc.

Key characteristics:
- VideoMAE-base  (MCG-NJU/videomae-base):                16 frames, 768-dim
- VideoMAE-large (MCG-NJU/videomae-large):               16 frames, 1024-dim
- VideoMAE-base  finetuned-kinetics (classification):    16 frames, 400 classes
- TimeSformer    (facebook/timesformer-base-finetuned-k400): 8 frames, 400 classes

Install:
    pip install 'sheaf-serve[video]'

Usage::

    spec = ModelSpec(
        name="videomae",
        model_type=ModelType.VIDEO,
        backend="videomae",
        backend_kwargs={
            "model_name": "MCG-NJU/videomae-base",
            "task": "embedding",
            "device": "cuda",
        },
        resources=ResourceConfig(num_gpus=1),
    )
"""

from __future__ import annotations

import base64
import io
from typing import Any, Literal

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.video import VideoRequest, VideoResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("videomae")
class VideoMAEBackend(ModelBackend):
    """ModelBackend for video understanding via VideoMAE / TimeSformer.

    Uses AutoModel (embedding task) or AutoModelForVideoClassification
    (classification task) from HuggingFace transformers, so any
    AutoModel-compatible video model works.

    Args:
        model_name: HuggingFace model ID.
            Embedding:       "MCG-NJU/videomae-base" (768-dim, default)
            Classification:  "MCG-NJU/videomae-base-finetuned-kinetics"
                             "facebook/timesformer-base-finetuned-k400"
        task: "embedding" or "classification".
        device: "cpu", "cuda", or "mps".
        pooling: CLS token ("cls", default) or mean of patch tokens ("mean").
            Only applies to the "embedding" task.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        task: Literal["embedding", "classification"] = "embedding",
        device: str = "cpu",
        pooling: Literal["cls", "mean"] = "cls",
    ) -> None:
        self._model_name = model_name
        self._task = task
        self._device = device
        self._pooling = pooling
        self._model: Any = None
        self._processor: Any = None
        self._Image: Any = None  # PIL.Image — stored at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.VIDEO

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from transformers import (  # ty: ignore[unresolved-import]
                AutoImageProcessor,
                AutoModel,
                AutoModelForVideoClassification,
            )
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for VideoMAEBackend. "
                "Install with: pip install 'sheaf-serve[video]'"
            ) from e

        self._processor = AutoImageProcessor.from_pretrained(self._model_name)

        if self._task == "classification":
            self._model = AutoModelForVideoClassification.from_pretrained(
                self._model_name
            )
        else:
            self._model = AutoModel.from_pretrained(self._model_name)

        self._model = self._model.to(self._device)
        self._model.eval()
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, VideoRequest):
            raise TypeError(f"Expected VideoRequest, got {type(request)}")
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: VideoRequest) -> VideoResponse:
        import torch  # ty: ignore[unresolved-import]

        # Decode frames
        frames = [
            self._Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            for b64 in request.frames_b64
        ]

        inputs = self._processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        if request.task == "classification":
            return self._classification_response(request, outputs)
        return self._embedding_response(request, outputs)

    def _embedding_response(self, request: VideoRequest, outputs: Any) -> VideoResponse:

        hidden = outputs.last_hidden_state  # (1, num_patches + 1, dim)
        if request.pooling == "cls":
            emb = hidden[:, 0, :]  # CLS token
        else:
            emb = hidden[:, 1:, :].mean(dim=1)  # mean of patch tokens

        if request.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        emb_list: list[float] = emb[0].cpu().float().tolist()

        return VideoResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            task="embedding",
            embedding=emb_list,
            dim=len(emb_list),
        )

    def _classification_response(
        self, request: VideoRequest, outputs: Any
    ) -> VideoResponse:
        import torch  # ty: ignore[unresolved-import]

        logits = outputs.logits  # (1, num_classes)
        num_classes: int = logits.shape[-1]
        probs = torch.softmax(logits[0], dim=-1)

        id2label: dict[int, str] = getattr(
            self._model.config,
            "id2label",
            {i: str(i) for i in range(num_classes)},
        )

        # Return top-k sorted by score descending
        top_k = min(5, num_classes)
        top_scores, top_ids = torch.topk(probs, k=top_k)

        labels = [id2label.get(int(i), str(int(i))) for i in top_ids]
        scores = top_scores.cpu().float().tolist()

        return VideoResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            task="classification",
            labels=labels,
            scores=scores,
        )
