"""ViTPose backend for human pose estimation via HuggingFace transformers.

Requires: pip install "sheaf-serve[pose]"
Models: "usyd-community/vitpose-base-simple" (default),
        "usyd-community/vitpose-plus-base"

ViTPose is a top-down pose estimator: it takes a person bounding box and returns
keypoints within that crop. If no bboxes are supplied in the request, the full
image is used as a single-person crop.

Output: COCO-style 17-keypoint skeleton (or whatever the model's id2label provides).
Keypoint coordinates are in absolute pixel space.

PIL.Image is stored as an instance attribute at load() time so tests can inject
a mock without PIL installed (same pattern as DETRBackend and DINOv2Backend).
"""

from __future__ import annotations

import base64
import io
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.pose import PoseRequest, PoseResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("vitpose")
class ViTPoseBackend(ModelBackend):
    """ModelBackend for ViTPose human pose estimation.

    Args:
        model_name: HuggingFace model ID.
            "usyd-community/vitpose-base-simple"  (default, COCO 17 kpts)
            "usyd-community/vitpose-plus-base"
            "usyd-community/vitpose-base-coco-aic-mpii"
        device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        model_name: str = "usyd-community/vitpose-base-simple",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._Image: Any = None  # PIL.Image — injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.POSE

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from transformers import (  # ty: ignore[unresolved-import]
                AutoProcessor,
                VitPoseForPoseEstimation,
            )
        except ImportError as e:
            raise ImportError(
                "transformers and Pillow are required for the ViTPose backend. "
                "Install with: pip install 'sheaf-serve[pose]'"
            ) from e
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = VitPoseForPoseEstimation.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, PoseRequest):
            raise TypeError(f"Expected PoseRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: PoseRequest) -> PoseResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        img = self._Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert(
            "RGB"
        )
        img_width, img_height = img.size  # PIL: (W, H)

        # ViTPose expects boxes as [[[x1,y1,x2,y2], ...]] — wrapped per image.
        boxes_for_image = (
            request.bboxes if request.bboxes else [[0, 0, img_width, img_height]]
        )
        boxes = [boxes_for_image]

        inputs = self._processor(images=img, boxes=boxes, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_pose_estimation(outputs, boxes=boxes)

        # results[0]: list of dicts per person — {"keypoints": Tensor, "scores": Tensor}
        keypoint_names = self._keypoint_names()
        poses: list[list[list[float]]] = []
        for person in results[0]:
            kpts = person["keypoints"].cpu().tolist()  # [[x, y], ...]
            scores = person["scores"].cpu().tolist()  # [s, ...]
            poses.append([[x, y, s] for (x, y), s in zip(kpts, scores)])

        return PoseResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            poses=poses,
            keypoint_names=keypoint_names,
            width=img_width,
            height=img_height,
        )

    def _keypoint_names(self) -> list[str]:
        """Return keypoint names from model config, falling back to numeric indices."""
        try:
            id2label: dict[int, str] = self._model.config.id2label
            return [id2label[i] for i in range(len(id2label))]
        except (AttributeError, KeyError):
            return []
