"""Stub backends for testing — no test-framework dependency.

Kept in a separate module so cloudpickle can import this file in Ray
worker processes (which don't have pytest installed).
"""

from __future__ import annotations

import base64

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.depth import DepthRequest, DepthResponse
from sheaf.api.detection import DetectionRequest, DetectionResponse
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.api.molecular import MolecularRequest, MolecularResponse
from sheaf.api.segmentation import SegmentationRequest, SegmentationResponse
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("_smoke_ts")
class SmokeTimeSeriesBackend(ModelBackend):
    """Echo backend: returns 0.42 for every horizon step."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, TimeSeriesRequest)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[0.42] * request.horizon,
        )


@register_backend("_smoke_ts_registry")
class SmokeTimeSeriesRegistryBackend(ModelBackend):
    """Like SmokeTimeSeriesBackend but deployed via string registry lookup.

    Used to verify that SHEAF_EXTRA_BACKENDS causes this module to be
    imported in Ray worker processes, making the registry entry available.
    """

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, TimeSeriesRequest)
        return TimeSeriesResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            horizon=request.horizon,
            frequency=request.frequency.value,
            mean=[0.99] * request.horizon,
        )


@register_backend("_smoke_error")
class ErrorTimeSeriesBackend(ModelBackend):
    """Backend that always raises — used to test service-boundary error handling."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.TIME_SERIES

    def predict(self, request: BaseRequest) -> BaseResponse:
        raise RuntimeError("backend exploded")


# ---------------------------------------------------------------------------
# New-modality stubs — no weights, fixed responses
# ---------------------------------------------------------------------------


@register_backend("_smoke_embedding")
class SmokeEmbeddingBackend(ModelBackend):
    """Returns fixed 3-dim L2-normalized embeddings."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.EMBEDDING

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, EmbeddingRequest)
        n = len(request.texts or request.images_b64 or [])
        return EmbeddingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=[[1.0, 0.0, 0.0]] * n,
            dim=3,
        )


@register_backend("_smoke_segmentation")
class SmokeSegmentationBackend(ModelBackend):
    """Returns a single all-ones 4×4 mask."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.SEGMENTATION

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, SegmentationRequest)
        mask = np.ones((4, 4), dtype=np.uint8)
        return SegmentationResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            masks_b64=[base64.b64encode(mask.tobytes()).decode()],
            scores=[0.9],
            height=4,
            width=4,
        )


@register_backend("_smoke_molecular")
class SmokeMolecularBackend(ModelBackend):
    """Returns fixed 3-dim embeddings for each input sequence."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.MOLECULAR

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, MolecularRequest)
        n = len(request.sequences)
        return MolecularResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=[[1.0, 0.0, 0.0]] * n,
            dim=3,
        )


@register_backend("_smoke_depth")
class SmokeDepthBackend(ModelBackend):
    """Returns a constant 0.5 depth map at 4×4 resolution."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.DEPTH

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, DepthRequest)
        depth = np.full((4, 4), 0.5, dtype=np.float32)
        return DepthResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            depth_b64=base64.b64encode(depth.tobytes()).decode(),
            height=4,
            width=4,
            min_depth=0.5,
            max_depth=0.5,
        )


@register_backend("_smoke_detection")
class SmokeDetectionBackend(ModelBackend):
    """Returns one fixed bounding box detection."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.DETECTION

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, DetectionRequest)
        return DetectionResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            boxes=[[10.0, 20.0, 100.0, 200.0]],
            scores=[0.9],
            labels=["cat"],
            width=640,
            height=480,
        )
