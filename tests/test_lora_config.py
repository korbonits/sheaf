"""Tests for LoRAAdapter / LoRAConfig + ModelSpec.lora wiring."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sheaf.api.base import ModelType
from sheaf.lora import LoRAAdapter, LoRAConfig
from sheaf.scheduling.batch import BatchPolicy
from sheaf.spec import ModelSpec


class TestLoRAAdapter:
    def test_minimal(self) -> None:
        a = LoRAAdapter(source="/tmp/foo.safetensors")
        assert a.source == "/tmp/foo.safetensors"
        assert a.weight == 1.0

    def test_custom_weight(self) -> None:
        a = LoRAAdapter(source="hf:org/repo", weight=0.5)
        assert a.weight == 0.5

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoRAAdapter(source="/tmp/x", weight=-0.1)

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoRAAdapter(source="")


class TestLoRAConfig:
    def test_empty_default(self) -> None:
        cfg = LoRAConfig()
        assert cfg.adapters == {}
        assert cfg.default is None

    def test_with_adapters(self) -> None:
        cfg = LoRAConfig(
            adapters={
                "sketch": LoRAAdapter(source="/tmp/sketch.safetensors"),
                "watercolor": LoRAAdapter(source="hf:user/watercolor", weight=0.7),
            },
            default="sketch",
        )
        assert set(cfg.adapters) == {"sketch", "watercolor"}
        assert cfg.adapters["watercolor"].weight == 0.7
        assert cfg.default == "sketch"

    def test_default_must_be_in_adapters(self) -> None:
        with pytest.raises(ValidationError) as exc:
            LoRAConfig(
                adapters={"sketch": LoRAAdapter(source="/tmp/sketch.safetensors")},
                default="watercolor",
            )
        assert "watercolor" in str(exc.value)

    def test_default_none_with_empty_adapters_ok(self) -> None:
        # No adapters configured, no default — valid (no-LoRA deployment).
        cfg = LoRAConfig()
        assert cfg.default is None


class TestModelSpecLoRA:
    def test_lora_default_is_none(self) -> None:
        spec = ModelSpec(
            name="m",
            model_type=ModelType.DIFFUSION,
            backend="flux",
        )
        assert spec.lora is None

    def test_lora_attaches_to_spec(self) -> None:
        spec = ModelSpec(
            name="flux-loras",
            model_type=ModelType.DIFFUSION,
            backend="flux",
            lora=LoRAConfig(
                adapters={"sketch": LoRAAdapter(source="/tmp/sketch.safetensors")},
                default="sketch",
            ),
        )
        assert spec.lora is not None
        assert spec.lora.default == "sketch"
        assert "sketch" in spec.lora.adapters

    def test_lora_invalid_default_propagates(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(
                name="bad",
                model_type=ModelType.DIFFUSION,
                backend="flux",
                lora=LoRAConfig(
                    adapters={"sketch": LoRAAdapter(source="/tmp/sketch")},
                    default="missing",
                ),
            )

    def test_lora_with_bucket_by_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ModelSpec(
                name="conflict",
                model_type=ModelType.DIFFUSION,
                backend="flux",
                batch_policy=BatchPolicy(bucket_by="height"),
                lora=LoRAConfig(
                    adapters={"sketch": LoRAAdapter(source="/tmp/sketch")},
                ),
            )

    def test_lora_with_bucket_by_none_ok(self) -> None:
        spec = ModelSpec(
            name="ok",
            model_type=ModelType.DIFFUSION,
            backend="flux",
            batch_policy=BatchPolicy(max_batch_size=8),
            lora=LoRAConfig(
                adapters={"sketch": LoRAAdapter(source="/tmp/sketch")},
            ),
        )
        assert spec.lora is not None
        assert spec.batch_policy.bucket_by is None
