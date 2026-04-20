"""Discriminated union over every supported request type.

Used by `sheaf.server` (FastAPI body parsing for /predict and /stream) and by
`sheaf.batch.runner` (per-row validation before Ray Data map_batches).  Kept in
its own module so batch workloads can validate input rows without importing
`sheaf.server`, which pulls in the full Ray Serve deployment surface.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from sheaf.api.audio import AudioRequest, TTSRequest
from sheaf.api.audio_generation import AudioGenerationRequest
from sheaf.api.depth import DepthRequest
from sheaf.api.detection import DetectionRequest
from sheaf.api.diffusion import DiffusionRequest
from sheaf.api.embedding import EmbeddingRequest
from sheaf.api.genomic import GenomicRequest
from sheaf.api.materials import MaterialsRequest
from sheaf.api.molecular import MolecularRequest
from sheaf.api.multimodal_embedding import MultimodalEmbeddingRequest
from sheaf.api.multimodal_generation import MultimodalGenerationRequest
from sheaf.api.optical_flow import OpticalFlowRequest
from sheaf.api.point_cloud import PointCloudRequest
from sheaf.api.pose import PoseRequest
from sheaf.api.satellite import SatelliteRequest
from sheaf.api.segmentation import SegmentationRequest
from sheaf.api.small_molecule import SmallMoleculeRequest
from sheaf.api.tabular import TabularRequest
from sheaf.api.time_series import TimeSeriesRequest
from sheaf.api.video import VideoRequest
from sheaf.api.weather import WeatherRequest

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

__all__ = ["AnyRequest"]
