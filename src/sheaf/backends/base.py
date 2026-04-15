"""Abstract base class for model backends."""

import asyncio
from abc import ABC, abstractmethod

from sheaf.api.base import BaseRequest, BaseResponse


class ModelBackend(ABC):
    """Pluggable model backend.

    Implement this to add a new model to Sheaf. The backend owns:
    - Model loading and initialization
    - Preprocessing raw inputs
    - Running inference
    - Postprocessing outputs into a typed response
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize any required state."""
        ...

    @abstractmethod
    def predict(self, request: BaseRequest) -> BaseResponse:
        """Run inference for a single request."""
        ...

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        """Run inference over a batch.

        Default implementation runs requests sequentially.
        Override to implement model-type-aware batching.
        """
        return [self.predict(r) for r in requests]

    async def async_predict(self, request: BaseRequest) -> BaseResponse:
        """Async wrapper around predict(); runs in a thread executor.

        Override for backends with native async inference.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, request)

    async def async_batch_predict(
        self, requests: list[BaseRequest]
    ) -> list[BaseResponse]:
        """Async wrapper around batch_predict(); runs in a thread executor.

        Override for backends with native async inference.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.batch_predict, requests)

    @property
    @abstractmethod
    def model_type(self) -> str:
        """The ModelType this backend serves."""
        ...
