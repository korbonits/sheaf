"""Abstract base class for model backends."""

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

    @property
    @abstractmethod
    def model_type(self) -> str:
        """The ModelType this backend serves."""
        ...
