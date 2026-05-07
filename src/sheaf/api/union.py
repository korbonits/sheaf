"""Discriminated unions over every supported request and response type.

Used by `sheaf.server` (FastAPI body parsing for /predict and /stream) and by
`sheaf.batch.runner` (per-row validation before Ray Data map_batches).  Kept in
its own module so batch workloads can validate input rows without importing
`sheaf.server`, which pulls in the full Ray Serve deployment surface.

`AnyResponse` mirrors `AnyRequest` and is what `sheaf.client.SheafClient` decodes
predict() responses into so callers get the correctly-typed response object back.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from sheaf.api.audio import AudioRequest, AudioResponse, TTSRequest, TTSResponse
from sheaf.api.audio_generation import AudioGenerationRequest, AudioGenerationResponse
from sheaf.api.depth import DepthRequest, DepthResponse
from sheaf.api.detection import DetectionRequest, DetectionResponse
from sheaf.api.diffusion import DiffusionRequest, DiffusionResponse
from sheaf.api.embedding import EmbeddingRequest, EmbeddingResponse
from sheaf.api.genomic import GenomicRequest, GenomicResponse
from sheaf.api.materials import MaterialsRequest, MaterialsResponse
from sheaf.api.molecular import MolecularRequest, MolecularResponse
from sheaf.api.multimodal_embedding import (
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
)
from sheaf.api.multimodal_generation import (
    MultimodalGenerationRequest,
    MultimodalGenerationResponse,
)
from sheaf.api.optical_flow import OpticalFlowRequest, OpticalFlowResponse
from sheaf.api.point_cloud import PointCloudRequest, PointCloudResponse
from sheaf.api.pose import PoseRequest, PoseResponse
from sheaf.api.satellite import SatelliteRequest, SatelliteResponse
from sheaf.api.segmentation import SegmentationRequest, SegmentationResponse
from sheaf.api.small_molecule import SmallMoleculeRequest, SmallMoleculeResponse
from sheaf.api.tabular import TabularRequest, TabularResponse
from sheaf.api.time_series import TimeSeriesRequest, TimeSeriesResponse
from sheaf.api.video import VideoRequest, VideoResponse
from sheaf.api.weather import WeatherRequest, WeatherResponse

AnyRequest = Annotated[
    TimeSeriesRequest
    | TabularRequest
    | AudioRequest
    | AudioGenerationRequest
    | TTSRequest
    | EmbeddingRequest
    | SegmentationRequest
    | MolecularRequest
    | GenomicRequest
    | MaterialsRequest
    | SmallMoleculeRequest
    | DepthRequest
    | DetectionRequest
    | WeatherRequest
    | SatelliteRequest
    | MultimodalEmbeddingRequest
    | DiffusionRequest
    | VideoRequest
    | PoseRequest
    | OpticalFlowRequest
    | MultimodalGenerationRequest
    | PointCloudRequest,
    Field(discriminator="model_type"),
]

AnyResponse = Annotated[
    TimeSeriesResponse
    | TabularResponse
    | AudioResponse
    | AudioGenerationResponse
    | TTSResponse
    | EmbeddingResponse
    | SegmentationResponse
    | MolecularResponse
    | GenomicResponse
    | MaterialsResponse
    | SmallMoleculeResponse
    | DepthResponse
    | DetectionResponse
    | WeatherResponse
    | SatelliteResponse
    | MultimodalEmbeddingResponse
    | DiffusionResponse
    | VideoResponse
    | PoseResponse
    | OpticalFlowResponse
    | MultimodalGenerationResponse
    | PointCloudResponse,
    Field(discriminator="model_type"),
]

__all__ = ["AnyRequest", "AnyResponse"]
