"""Tests for the LoRA hooks on the ModelBackend ABC."""

from __future__ import annotations

import pytest

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.backends.base import ModelBackend
from sheaf.lora import LoRAAdapter


class _NoOpBackend(ModelBackend):
    """Minimal concrete backend so we can instantiate the ABC."""

    def load(self) -> None:
        pass

    def predict(self, request: BaseRequest) -> BaseResponse:
        return BaseResponse(
            request_id=request.request_id,
            model_type=ModelType.DIFFUSION,
            model_name=request.model_name,
        )

    @property
    def model_type(self) -> str:
        return ModelType.DIFFUSION.value


class TestLoRADefaults:
    def test_supports_lora_false_by_default(self) -> None:
        b = _NoOpBackend()
        assert b.supports_lora() is False

    def test_load_adapters_raises(self) -> None:
        b = _NoOpBackend()
        with pytest.raises(NotImplementedError, match="LoRA"):
            b.load_adapters({"x": LoRAAdapter(source="/tmp/x")})

    def test_set_active_adapters_raises(self) -> None:
        b = _NoOpBackend()
        with pytest.raises(NotImplementedError, match="LoRA"):
            b.set_active_adapters(["x"], [1.0])
