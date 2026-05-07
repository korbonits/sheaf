"""Tests for LoRA support on FluxBackend.

Mocks the diffusers pipeline like ``test_flux_backend.py`` and asserts that
``load_adapters`` / ``set_active_adapters`` translate to the expected
``pipeline.load_lora_weights`` / ``pipeline.set_adapters`` calls.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from sheaf.backends.flux import _parse_lora_source
from sheaf.lora import LoRAAdapter


def _make_fake_pipeline() -> MagicMock:
    instance = MagicMock()
    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value.to.return_value = instance
    return pipeline_cls, instance


def _make_fake_torch() -> MagicMock:
    torch = MagicMock()
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    return torch


def _loaded_backend() -> tuple:
    from sheaf.backends.flux import FluxBackend

    pipeline_cls, instance = _make_fake_pipeline()
    fake_diffusers = MagicMock()
    fake_diffusers.FluxPipeline = pipeline_cls
    fake_torch = _make_fake_torch()

    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        backend = FluxBackend(device="cpu")
        backend.load()

    return backend, instance


class TestParseLoraSource:
    def test_local_path(self) -> None:
        assert _parse_lora_source("/tmp/sketch.safetensors") == (
            "/tmp/sketch.safetensors",
            None,
        )

    def test_relative_path(self) -> None:
        assert _parse_lora_source("./loras/sketch") == ("./loras/sketch", None)

    def test_hf_repo_only(self) -> None:
        assert _parse_lora_source("hf:user/repo") == ("user/repo", None)

    def test_hf_repo_with_weight_name(self) -> None:
        assert _parse_lora_source("hf:user/repo:weight.safetensors") == (
            "user/repo",
            "weight.safetensors",
        )


class TestFluxLoRAHooks:
    def test_supports_lora(self) -> None:
        backend, _ = _loaded_backend()
        assert backend.supports_lora() is True

    def test_load_adapters_local_path(self) -> None:
        backend, instance = _loaded_backend()
        backend.load_adapters({"sketch": LoRAAdapter(source="/tmp/sketch.safetensors")})
        instance.load_lora_weights.assert_called_once_with(
            "/tmp/sketch.safetensors", adapter_name="sketch"
        )

    def test_load_adapters_hf_repo(self) -> None:
        backend, instance = _loaded_backend()
        backend.load_adapters({"watercolor": LoRAAdapter(source="hf:user/repo")})
        instance.load_lora_weights.assert_called_once_with(
            "user/repo", adapter_name="watercolor"
        )

    def test_load_adapters_hf_repo_with_weight_name(self) -> None:
        backend, instance = _loaded_backend()
        backend.load_adapters(
            {"pixar": LoRAAdapter(source="hf:user/repo:pixar.safetensors")}
        )
        instance.load_lora_weights.assert_called_once_with(
            "user/repo",
            adapter_name="pixar",
            weight_name="pixar.safetensors",
        )

    def test_load_adapters_multiple(self) -> None:
        backend, instance = _loaded_backend()
        backend.load_adapters(
            {
                "sketch": LoRAAdapter(source="/tmp/sketch.safetensors"),
                "watercolor": LoRAAdapter(source="hf:user/wc"),
            }
        )
        assert instance.load_lora_weights.call_count == 2

    def test_set_active_adapters(self) -> None:
        backend, instance = _loaded_backend()
        backend.set_active_adapters(["sketch"], [0.7])
        instance.enable_lora.assert_called_once()
        instance.set_adapters.assert_called_once_with(["sketch"], adapter_weights=[0.7])

    def test_set_active_adapters_fusion(self) -> None:
        backend, instance = _loaded_backend()
        backend.set_active_adapters(["a", "b"], [1.0, 0.5])
        instance.enable_lora.assert_called_once()
        instance.set_adapters.assert_called_once_with(
            ["a", "b"], adapter_weights=[1.0, 0.5]
        )

    def test_set_active_adapters_empty_disables_lora(self) -> None:
        """Empty names → pipeline.disable_lora(), not set_adapters([], [])."""
        backend, instance = _loaded_backend()
        backend.set_active_adapters([], [])
        instance.disable_lora.assert_called_once()
        instance.set_adapters.assert_not_called()
        instance.enable_lora.assert_not_called()

    def test_set_active_adapters_length_mismatch(self) -> None:
        backend, _ = _loaded_backend()
        with pytest.raises(ValueError, match="length"):
            backend.set_active_adapters(["a", "b"], [1.0])

    def test_load_adapters_before_load_raises(self) -> None:
        from sheaf.backends.flux import FluxBackend

        backend = FluxBackend(device="cpu")
        with pytest.raises(RuntimeError, match="Backend not loaded"):
            backend.load_adapters({"x": LoRAAdapter(source="/tmp/x")})

    def test_set_active_before_load_raises(self) -> None:
        from sheaf.backends.flux import FluxBackend

        backend = FluxBackend(device="cpu")
        with pytest.raises(RuntimeError, match="Backend not loaded"):
            backend.set_active_adapters(["x"], [1.0])
