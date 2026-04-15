"""Batching policies for model-type-aware request scheduling."""

from pydantic import BaseModel, Field


class BatchPolicy(BaseModel):
    """Controls how requests are batched before hitting the model backend.

    max_batch_size: hard cap on requests per batch
    timeout_ms: max time to wait for a full batch before flushing
    bucket_by: field name to bucket on before batching (e.g. "horizon" for
               time series, so variable-length forecasts don't get mixed)
    """

    max_batch_size: int = Field(default=32, gt=0)
    timeout_ms: int = Field(default=50, gt=0)
    bucket_by: str | None = None
