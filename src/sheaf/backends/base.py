"""Abstract base class for model backends."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from sheaf.api.base import BaseRequest, BaseResponse

if TYPE_CHECKING:
    from sheaf.lora import LoRAAdapter


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

    async def stream_predict(
        self, request: BaseRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream inference events for a single request.

        Default implementation runs predict() and yields a single result event.
        Override in backends that support chunked or progressive output
        (e.g. diffusion step-end callbacks, TTS phoneme tokens).

        Yields:
            Event dicts with at least ``"type"`` and ``"done"`` keys:

            - Progress: ``{"type": "progress", "step": N, "total_steps": N}``
            - Result:   ``{"type": "result", "done": true, ...response_fields}``
        """
        result = await self.async_predict(request)
        yield {"type": "result", "done": True, **result.model_dump(mode="json")}

    @property
    @abstractmethod
    def model_type(self) -> str:
        """The ModelType this backend serves."""
        ...

    # ------------------------------------------------------------------
    # LoRA adapter multiplexing — opt-in
    # ------------------------------------------------------------------

    def supports_lora(self) -> bool:
        """Whether this backend can host LoRA adapters.

        Default ``False``.  Override and return ``True`` in backends that
        implement :meth:`load_adapters` and :meth:`set_active_adapters`.
        """
        return False

    def load_adapters(self, adapters: dict[str, "LoRAAdapter"]) -> None:
        """Load a registry of named LoRA adapters into the backend.

        Called once by ``_SheafDeployment.__init__`` (or the Modal startup
        path) after ``load()``.  Implementations should iterate the mapping
        and load each adapter under its dict key as the adapter name.

        Args:
            adapters: Mapping of adapter name → :class:`sheaf.lora.LoRAAdapter`.
                Adapter sources may be local paths or ``hf:`` references; the
                backend is responsible for parsing the source string.

        Raises:
            NotImplementedError: Default — backends that return ``True`` from
                :meth:`supports_lora` must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support LoRA adapters"
        )

    def set_active_adapters(self, names: list[str], weights: list[float]) -> None:
        """Activate a subset of previously loaded adapters with given weights.

        Called per sub-batch (after ``bucket_by_adapter`` grouping) before
        ``async_batch_predict``.  ``names`` is empty for a no-LoRA call.

        Args:
            names:   Adapter names to activate.  Each must be a key passed to
                a prior :meth:`load_adapters` call.  Empty list means
                "deactivate all adapters" (run with the base model only).
            weights: Per-adapter weights, parallel to ``names``.  Length must
                match ``names``.

        Raises:
            NotImplementedError: Default — backends that return ``True`` from
                :meth:`supports_lora` must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support LoRA adapters"
        )
