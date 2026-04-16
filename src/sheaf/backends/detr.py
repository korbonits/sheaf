"""DETR / RT-DETR backend for object detection via HuggingFace transformers.

Requires: pip install "sheaf-serve[vision]"
Library: transformers (https://huggingface.co/docs/transformers/model_doc/detr)

Supported architectures (examples):
  "facebook/detr-resnet-50"      — DETR ResNet-50, 91 COCO classes (default)
  "facebook/detr-resnet-101"     — DETR ResNet-101, higher quality
  "PekingU/rtdetr_r50vd"         — RT-DETR ResNet-50, real-time speed
  "PekingU/rtdetr_r101vd"        — RT-DETR ResNet-101

Any model loadable via AutoModelForObjectDetection is supported.

Boxes are returned in absolute pixel coordinates [x1, y1, x2, y2].
Detections below ``threshold`` are filtered out by the processor's
post_process_object_detection step.

PIL.Image is stored as an instance attribute at load() time so the
dependency stays lazy and tests can inject a mock without PIL installed.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.detection import DetectionRequest, DetectionResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("detr")
class DETRBackend(ModelBackend):
    """ModelBackend for DETR / RT-DETR object detection.

    Detects objects in a single image and returns bounding boxes, class
    labels, and confidence scores for all detections above ``threshold``.

    Args:
        model_name: HuggingFace model ID.
            "facebook/detr-resnet-50"   (default, 91 COCO classes)
            "facebook/detr-resnet-101"
            "PekingU/rtdetr_r50vd"
            "PekingU/rtdetr_r101vd"
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._Image: Any = None  # PIL.Image, injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.DETECTION

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from transformers import (  # ty: ignore[unresolved-import]
                AutoImageProcessor,
                AutoModelForObjectDetection,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the DETR backend. "
                "Install it with: pip install 'sheaf-serve[vision]'"
            ) from e
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModelForObjectDetection.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, DetectionRequest):
            raise TypeError(f"Expected DetectionRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: DetectionRequest) -> DetectionResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        img = self._Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert(
            "RGB"
        )
        img_width, img_height = img.size  # PIL: (W, H)

        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            threshold=request.threshold,
            target_sizes=[(img_height, img_width)],
        )[0]

        boxes: list[list[float]] = results["boxes"].cpu().tolist()
        scores: list[float] = results["scores"].cpu().tolist()
        labels: list[str] = [
            self._model.config.id2label[label_id.item()]
            for label_id in results["labels"]
        ]

        return DetectionResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            boxes=boxes,
            scores=scores,
            labels=labels,
            width=img_width,
            height=img_height,
        )
