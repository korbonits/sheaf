"""RAFT optical flow backend via torchvision.

Requires: pip install "sheaf-serve[optical-flow]"
Models: "raft_large" (default), "raft_small"

RAFT (Recurrent All-Pairs Field Transforms) estimates dense optical flow between
two consecutive frames. torchvision ships pretrained weights; no HuggingFace Hub
login required.

The flow field is returned as a base64-encoded float32 array of shape (H, W, 2)
where the last dimension is (dx, dy) — pixel displacements from frame1 to frame2.

RAFT's preprocessing pads inputs to multiples of 8 pixels; the output is cropped
back to the original frame dimensions before encoding.

self._transforms and self._Image are set at load() time for test injectability,
following the same pattern as DINOv2Backend and ViTPoseBackend.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.optical_flow import OpticalFlowRequest, OpticalFlowResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_VALID_MODELS = {"raft_large", "raft_small"}


@register_backend("raft")
class RAFTBackend(ModelBackend):
    """ModelBackend for RAFT optical flow (torchvision).

    Args:
        model_name: "raft_large" (default, higher quality) or "raft_small"
            (faster, lower memory).
        device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        model_name: str = "raft_large",
        device: str = "cpu",
    ) -> None:
        if model_name not in _VALID_MODELS:
            raise ValueError(
                f"model_name must be one of {_VALID_MODELS}, got {model_name!r}"
            )
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._transforms: Any = None
        self._Image: Any = None  # PIL.Image — injected at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.OPTICAL_FLOW

    def load(self) -> None:
        try:
            from PIL import Image as _Image  # ty: ignore[unresolved-import]
            from torchvision.models.optical_flow import (  # ty: ignore[unresolved-import]
                Raft_Large_Weights,
                Raft_Small_Weights,
                raft_large,
                raft_small,
            )
        except ImportError as e:
            raise ImportError(
                "torchvision and Pillow are required for the RAFT backend. "
                "Install with: pip install 'sheaf-serve[optical-flow]'"
            ) from e
        if self._model_name == "raft_large":
            weights = Raft_Large_Weights.DEFAULT
            self._model = raft_large(weights=weights)
        else:
            weights = Raft_Small_Weights.DEFAULT
            self._model = raft_small(weights=weights)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._transforms = weights.transforms()
        self._Image = _Image

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, OpticalFlowRequest):
            raise TypeError(f"Expected OpticalFlowRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: OpticalFlowRequest) -> OpticalFlowResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        img1 = self._Image.open(
            io.BytesIO(base64.b64decode(request.frame1_b64))
        ).convert("RGB")
        img2 = self._Image.open(
            io.BytesIO(base64.b64decode(request.frame2_b64))
        ).convert("RGB")

        img1_np = np.array(img1)  # (H, W, 3) uint8
        img2_np = np.array(img2)
        orig_h, orig_w = img1_np.shape[:2]

        # Build (1, 3, H, W) uint8 tensors for RAFT preprocessing.
        img1_t = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0)
        img2_t = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0)

        # Normalize + pad to multiples of 8.
        img1_t, img2_t = self._transforms(img1_t, img2_t)
        img1_t = img1_t.to(self._device)
        img2_t = img2_t.to(self._device)

        with torch.no_grad():
            flow_predictions = self._model(img1_t, img2_t)

        # flow_predictions[-1]: (1, 2, H_padded, W_padded); take batch index 0.
        flow_np: np.ndarray = flow_predictions[-1][0].cpu().numpy()  # (2, H_pad, W_pad)
        flow_np = flow_np[:, :orig_h, :orig_w]  # crop padding
        flow_np = np.transpose(flow_np, (1, 2, 0))  # (H, W, 2)

        flow_b64 = base64.b64encode(flow_np.astype(np.float32).tobytes()).decode()

        return OpticalFlowResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            flow_b64=flow_b64,
            width=orig_w,
            height=orig_h,
        )
