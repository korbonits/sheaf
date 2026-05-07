"""Tests for LoRA support on SDXLBackend."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sheaf.lora import LoRAAdapter


def _make_fake_diffusers() -> ModuleType:
    img2img_cls = MagicMock()
    inpaint_cls = MagicMock()
    img2img_instance = MagicMock()
    inpaint_instance = MagicMock()
    img2img_cls.from_pretrained.return_value.to.return_value = img2img_instance
    inpaint_cls.from_pretrained.return_value.to.return_value = inpaint_instance
    mod = ModuleType("diffusers")
    mod.StableDiffusionXLImg2ImgPipeline = img2img_cls  # type: ignore[attr-defined]
    mod.StableDiffusionXLInpaintPipeline = inpaint_cls  # type: ignore[attr-defined]
    mod._img2img_instance = img2img_instance  # type: ignore[attr-defined]
    mod._inpaint_instance = inpaint_instance  # type: ignore[attr-defined]
    return mod


def _make_fake_torch() -> MagicMock:
    t = MagicMock()
    t.float32 = "float32"
    t.float16 = "float16"
    t.bfloat16 = "bfloat16"
    return t


def _make_fake_pil() -> ModuleType:
    pil_mod = ModuleType("PIL")
    pil_mod.Image = MagicMock()  # type: ignore[attr-defined]
    return pil_mod


def _loaded(mode: str = "img2img") -> tuple:
    from sheaf.backends.sdxl import SDXLBackend

    diffusers = _make_fake_diffusers()
    backend = SDXLBackend(mode=mode, device="cpu")
    with patch.dict(
        sys.modules,
        {"diffusers": diffusers, "torch": _make_fake_torch(), "PIL": _make_fake_pil()},
    ):
        backend.load()

    instance = (
        diffusers._img2img_instance  # type: ignore[attr-defined]
        if mode == "img2img"
        else diffusers._inpaint_instance  # type: ignore[attr-defined]
    )
    return backend, instance


class TestSDXLLoRAHooks:
    def test_supports_lora(self) -> None:
        backend, _ = _loaded()
        assert backend.supports_lora() is True

    def test_load_adapters_local_path(self) -> None:
        backend, instance = _loaded()
        backend.load_adapters({"pixar": LoRAAdapter(source="/loras/pixar.safetensors")})
        instance.load_lora_weights.assert_called_once_with(
            "/loras/pixar.safetensors", adapter_name="pixar"
        )

    def test_load_adapters_hf_with_weight_name(self) -> None:
        backend, instance = _loaded()
        backend.load_adapters(
            {"pixar": LoRAAdapter(source="hf:user/repo:pixar.safetensors")}
        )
        instance.load_lora_weights.assert_called_once_with(
            "user/repo",
            adapter_name="pixar",
            weight_name="pixar.safetensors",
        )

    def test_set_active_adapters(self) -> None:
        backend, instance = _loaded()
        backend.set_active_adapters(["pixar"], [0.6])
        instance.enable_lora.assert_called_once()
        instance.set_adapters.assert_called_once_with(["pixar"], adapter_weights=[0.6])

    def test_set_active_adapters_fusion(self) -> None:
        backend, instance = _loaded()
        backend.set_active_adapters(["a", "b"], [0.5, 0.7])
        instance.enable_lora.assert_called_once()
        instance.set_adapters.assert_called_once_with(
            ["a", "b"], adapter_weights=[0.5, 0.7]
        )

    def test_set_active_adapters_empty_disables_lora(self) -> None:
        backend, instance = _loaded()
        backend.set_active_adapters([], [])
        instance.disable_lora.assert_called_once()
        instance.set_adapters.assert_not_called()
        instance.enable_lora.assert_not_called()

    def test_inpaint_mode_uses_inpaint_pipeline(self) -> None:
        backend, instance = _loaded(mode="inpaint")
        backend.load_adapters({"x": LoRAAdapter(source="/tmp/x")})
        instance.load_lora_weights.assert_called_once()

    def test_set_length_mismatch_rejected(self) -> None:
        backend, _ = _loaded()
        with pytest.raises(ValueError, match="length"):
            backend.set_active_adapters(["a", "b"], [1.0])

    def test_load_before_init_raises(self) -> None:
        from sheaf.backends.sdxl import SDXLBackend

        backend = SDXLBackend(device="cpu")
        with pytest.raises(RuntimeError, match="Backend not loaded"):
            backend.load_adapters({"x": LoRAAdapter(source="/tmp/x")})
