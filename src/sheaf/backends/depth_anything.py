"""Depth Anything v2 backend for monocular depth estimation.

Requires: pip install "sheaf-serve[vision]"
Library: transformers (https://huggingface.co/depth-anything)

Supported models (HuggingFace Hub):
  "depth-anything/Depth-Anything-V2-Small-hf"  — fastest (default)
  "depth-anything/Depth-Anything-V2-Base-hf"   — balanced
  "depth-anything/Depth-Anything-V2-Large-hf"  — highest quality

The depth map is returned at the model's native output resolution as a
base64-encoded float32 array.  Set normalize=True (default) to rescale to
[0, 1]; set normalize=False to get raw relative depth values.

PIL.Image is stored as an instance attribute at load() time so the
dependency stays lazy and tests can inject a mock without PIL installed.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.depth import DepthRequest, DepthResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("depth-anything")
class DepthAnythingBackend(ModelBackend):
    """ModelBackend for Depth Anything v2 monocular depth estimation.

    Returns a float32 depth map at the model's native output resolution.
    Depth values are relative (not metric); use normalize=True to map them
    to [0, 1] for display or downstream use.

    Args:
        model_name: HuggingFace model ID. Options:
            "depth-anything/Depth-Anything-V2-Small-hf"  (default)
            "depth-anything/Depth-Anything-V2-Base-hf"
            "depth-anything/Depth-Anything-V2-Large-hf"
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._Image: Any = None  # PIL.Image, injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.DEPTH

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from transformers import (  # ty: ignore[unresolved-import]
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the DepthAnything backend. "
                "Install it with: pip install 'sheaf-serve[vision]'"
            ) from e
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, DepthRequest):
            raise TypeError(f"Expected DepthRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: DepthRequest) -> DepthResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        img = self._Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert(
            "RGB"
        )

        inputs = self._processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # predicted_depth: (1, H, W) → squeeze to (H, W)
        depth_np: np.ndarray = outputs.predicted_depth.squeeze().cpu().numpy()

        min_depth = float(depth_np.min())
        max_depth = float(depth_np.max())

        if request.normalize:
            range_d = max_depth - min_depth
            if range_d > 0:
                depth_np = (depth_np - min_depth) / range_d
            else:
                depth_np = np.zeros_like(depth_np)

        height, width = depth_np.shape
        depth_b64 = base64.b64encode(depth_np.astype(np.float32).tobytes()).decode()

        return DepthResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            depth_b64=depth_b64,
            height=height,
            width=width,
            min_depth=min_depth,
            max_depth=max_depth,
        )
