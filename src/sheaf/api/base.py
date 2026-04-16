"""Base request/response contracts and model type registry."""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    TIME_SERIES = "time_series"
    TABULAR = "tabular"
    MOLECULAR = "molecular"
    GEOSPATIAL = "geospatial"
    DIFFUSION = "diffusion"
    NEURAL_OPERATOR = "neural_operator"
    AUDIO = "audio"
    TTS = "tts"
    EMBEDDING = "embedding"


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
