"""PointNet backend for 3D point cloud processing.

Requires: pip install "sheaf-serve[lidar]"

Implements the original PointNet architecture (Qi et al. 2017) in pure PyTorch
with no torch-geometric or C++ extensions required.  Supports two tasks:

  task="embed"    — returns the 1024-dim global max-pooled feature vector
                    (L2-normalised), suitable for retrieval and similarity search.
  task="classify" — returns the predicted class label and per-class softmax
                    scores over the 40 ModelNet40 categories (default) or a
                    custom class set matching the loaded checkpoint.

Architecture
------------
The _PointNetModel embedded here follows the standard PointNet classification
network: shared MLP (3→64→128→1024) applied point-wise via Conv1d + BatchNorm,
global max pooling to a 1024-dim descriptor, then a 3-layer classification head
(1024→512→256→num_classes) with dropout(0.3) and BatchNorm.

Checkpoint loading
------------------
Supply ``checkpoint_path`` pointing to a ``torch.save``-ed state dict or a dict
with key ``"model_state_dict"``.  If ``checkpoint_path`` is None, weights are
randomly initialised (useful for embedding without needing a trained classifier;
classification results will be meaningless).

Input format
------------
``PointCloudRequest.points_b64`` encodes a float32 array of shape (N, 3).
Points should be normalised to a unit sphere centred at the origin before
sending (subtract centroid, divide by max radius) — the same preprocessing used
when training standard PointNet checkpoints.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.point_cloud import PointCloudRequest, PointCloudResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

# ModelNet40 class names in the standard label order used by most PointNet
# checkpoints trained on ModelNet40.
_MODELNET40_CLASSES = (
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
)

_VALID_TASKS = {"embed", "classify"}


# ---------------------------------------------------------------------------
# PointNet architecture — pure PyTorch, no C++ extensions required
# ---------------------------------------------------------------------------


def _build_pointnet(num_classes: int) -> Any:
    """Build the PointNet classification network.

    Imported lazily inside load() so torch is not required at module import time.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _PointNetModel(nn.Module):
        def __init__(self, n_cls: int) -> None:
            super().__init__()
            # Point-wise shared MLP via 1-D convolution over the point dimension.
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            # Classification head.
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, n_cls)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
            self.drop = nn.Dropout(p=0.3)

        def forward(
            self,
            x: Any,  # (B, N, 3) float tensor
        ) -> tuple[Any, Any]:
            x = x.transpose(2, 1)  # → (B, 3, N)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            feat = torch.max(x, dim=2)[0]  # global max pool → (B, 1024)
            x = F.relu(self.bn4(self.fc1(feat)))
            x = F.relu(self.bn5(self.drop(self.fc2(x))))
            logits = self.fc3(x)  # (B, num_classes)
            return logits, feat

    return _PointNetModel(num_classes)


@register_backend("pointnet")
class PointNetBackend(ModelBackend):
    """ModelBackend for PointNet 3D point cloud classification and embedding.

    Args:
        checkpoint_path: Path to a ``torch.save``-ed state dict (or dict with
            key ``"model_state_dict"``).  If None, weights are randomly
            initialised — useful for generating embeddings without a trained
            classifier.
        num_classes: Number of output classes.  Must match the checkpoint.
            Default 40 (ModelNet40).
        label_names: Ordered list of class name strings, length == num_classes.
            Defaults to the standard ModelNet40 40-class list.
        device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        num_classes: int = 40,
        label_names: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._num_classes = num_classes
        self._label_names: list[str] = (
            label_names
            if label_names is not None
            else list(_MODELNET40_CLASSES[:num_classes])
        )
        self._device = device
        self._model: Any = None
        self._F: Any = None  # torch.nn.functional — stored at load() for testability

    @property
    def model_type(self) -> str:
        return ModelType.POINT_CLOUD

    def load(self) -> None:
        try:
            import torch  # ty: ignore[unresolved-import]
            import torch.nn.functional as _F  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "torch is required for the PointNet backend. "
                "Install with: pip install 'sheaf-serve[lidar]'"
            ) from e

        self._F = _F
        model = _build_pointnet(self._num_classes)

        if self._checkpoint_path is not None:
            ckpt = torch.load(self._checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)

        model = model.to(self._device)
        model.eval()
        self._model = model

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, PointCloudRequest):
            raise TypeError(f"Expected PointCloudRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: PointCloudRequest) -> PointCloudResponse:
        import torch  # ty: ignore[unresolved-import]

        F = self._F
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        pts = np.frombuffer(
            base64.b64decode(request.points_b64), dtype=np.float32
        ).reshape(request.n_points, 3)

        # Add batch dimension: (1, N, 3)
        x = torch.from_numpy(pts.copy()).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, feat = self._model(x)

        if request.task == "embed":
            emb = feat[0]  # (1024,)
            emb = F.normalize(emb, dim=0)
            return PointCloudResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                embedding=emb.cpu().tolist(),
            )

        # task == "classify"
        probs = F.softmax(logits[0], dim=0)
        scores = probs.cpu().tolist()
        top_idx = int(probs.argmax().item())
        label = (
            self._label_names[top_idx]
            if top_idx < len(self._label_names)
            else str(top_idx)
        )
        return PointCloudResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            label=label,
            scores=scores,
            label_names=self._label_names,
        )
