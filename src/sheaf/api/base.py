"""Base request/response contracts and model type registry."""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelType(StrEnum):
    TIME_SERIES = "time_series"
    TABULAR = "tabular"
    MOLECULAR = "molecular"
    GENOMIC = "genomic"
    MATERIALS = "materials"
    SMALL_MOLECULE = "small_molecule"
    GEOSPATIAL = "geospatial"
    WEATHER = "weather"
    DIFFUSION = "diffusion"
    VIDEO = "video"
    NEURAL_OPERATOR = "neural_operator"
    AUDIO = "audio"
    AUDIO_GENERATION = "audio_generation"
    TTS = "tts"
    EMBEDDING = "embedding"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    DETECTION = "detection"
    POSE = "pose"
    OPTICAL_FLOW = "optical_flow"
    MULTIMODAL_GENERATION = "multimodal_generation"
    POINT_CLOUD = "point_cloud"


class BaseRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    model_type: ModelType
    model_name: str

    model_config = {"arbitrary_types_allowed": True}


class BaseResponse(BaseModel):
    request_id: UUID
    model_type: ModelType
    model_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
