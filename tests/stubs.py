"""Stub backends for testing — no test-framework dependency.

Kept in a separate module so cloudpickle can import this file in Ray
worker processes (which don't have pytest installed).
"""

from __future__ import annotations

import base64

import numpy as np

from sheaf.api.audio_generation import AudioGenerationRequest, AudioGenerationResponse
from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.depth import DepthRequest, DepthResponse
from sheaf.api.detection import DetectionRequest, DetectionResponse
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.api.genomic import GenomicRequest, GenomicResponse
from sheaf.api.materials import MaterialsRequest, MaterialsResponse
from sheaf.api.molecular import MolecularRequest, MolecularResponse
from sheaf.api.multimodal_embedding import (
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
)
from sheaf.api.satellite import SatelliteRequest, SatelliteResponse
from sheaf.api.segmentation import SegmentationRequest, SegmentationResponse
from sheaf.api.small_molecule import SmallMoleculeRequest, SmallMoleculeResponse
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse
from sheaf.api.weather import WeatherRequest, WeatherResponse
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


@register_backend("_smoke_weather")
class SmokeWeatherBackend(ModelBackend):
    """Returns a single 6-hour forecast step with constant 0.0 fields."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.WEATHER

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, WeatherRequest)
        n_lat, n_lon = len(request.lat), len(request.lon)
        n_lev = len(request.pressure_levels)
        zeros_surf = base64.b64encode(
            np.zeros((n_lat, n_lon), dtype=np.float32).tobytes()
        ).decode()
        zeros_atmos = base64.b64encode(
            np.zeros((n_lev, n_lat, n_lon), dtype=np.float32).tobytes()
        ).decode()
        from datetime import datetime, timedelta

        current_dt = datetime.fromisoformat(request.current_time)
        surface_forecasts = [
            {k: zeros_surf for k in request.surface_vars}
            for _ in range(request.n_steps)
        ]
        atmospheric_forecasts = [
            {k: zeros_atmos for k in request.atmospheric_vars}
            for _ in range(request.n_steps)
        ]
        forecast_times = [
            (current_dt + timedelta(hours=6 * (i + 1))).isoformat()
            for i in range(request.n_steps)
        ]
        return WeatherResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            surface_forecasts=surface_forecasts,
            atmospheric_forecasts=atmospheric_forecasts,
            lat=request.lat,
            lon=request.lon,
            pressure_levels=request.pressure_levels,
            forecast_times=forecast_times,
            step_hours=6,
            n_steps=request.n_steps,
        )


@register_backend("_smoke_satellite")
class SmokeSatelliteBackend(ModelBackend):
    """Returns a fixed 4-dim all-zeros scene embedding."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.GEOSPATIAL

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, SatelliteRequest)
        return SatelliteResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embedding=[0.0, 0.0, 0.0, 0.0],
            dim=4,
            n_time=request.n_time,
        )


@register_backend("_smoke_genomic")
class SmokeGenomicBackend(ModelBackend):
    """Returns fixed 4-dim zero embeddings for each input sequence."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.GENOMIC

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, GenomicRequest)
        n = len(request.sequences)
        return GenomicResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=[[0.0, 0.0, 0.0, 0.0]] * n,
            dim=4,
        )


@register_backend("_smoke_materials")
class SmokeMaterialsBackend(ModelBackend):
    """Returns fixed energy=-42.0 eV and zero forces for any structure."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.MATERIALS

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, MaterialsRequest)
        n = len(request.atomic_numbers)
        forces_b64 = base64.b64encode(
            np.zeros((n, 3), dtype=np.float32).tobytes()
        ).decode()
        return MaterialsResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            energy=-42.0,
            forces_b64=forces_b64,
            n_atoms=n,
        )


@register_backend("_smoke_audio_generation")
class SmokeAudioGenerationBackend(ModelBackend):
    """Returns a minimal silent WAV for any generation prompt."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.AUDIO_GENERATION

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, AudioGenerationRequest)
        # 0.1s silent WAV at 32000 Hz
        n_samples = 3200
        from sheaf.backends._audio_utils import encode_wav

        silent = np.zeros(n_samples, dtype=np.float32)
        wav_bytes = encode_wav(silent, 32000)
        return AudioGenerationResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            audio_b64=base64.b64encode(wav_bytes).decode(),
            sampling_rate=32000,
            duration_s=n_samples / 32000,
        )


@register_backend("_smoke_multimodal")
class SmokeMultimodalEmbeddingBackend(ModelBackend):
    """Returns fixed 4-dim zero embeddings for each input item."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.MULTIMODAL_EMBEDDING

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, MultimodalEmbeddingRequest)
        n = request.n_items
        return MultimodalEmbeddingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=[[0.0, 0.0, 0.0, 0.0]] * n,
            dim=4,
            modality=request.modality,
        )


@register_backend("_smoke_small_molecule")
class SmokeSmallMoleculeBackend(ModelBackend):
    """Returns fixed 4-dim zero embeddings for each input SMILES."""

    def load(self) -> None:
        pass

    @property
    def model_type(self) -> str:
        return ModelType.SMALL_MOLECULE

    def predict(self, request: BaseRequest) -> BaseResponse:
        assert isinstance(request, SmallMoleculeRequest)
        n = len(request.smiles)
        return SmallMoleculeResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=[[0.0, 0.0, 0.0, 0.0]] * n,
            dim=4,
        )
