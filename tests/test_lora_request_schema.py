"""Tests for adapters / adapter_weights fields on FLUX + SDXL requests."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from sheaf.api.diffusion import DiffusionRequest
from sheaf.api.multimodal_generation import MultimodalGenerationRequest

_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


class TestDiffusionRequestAdapters:
    def test_defaults(self) -> None:
        r = DiffusionRequest(model_name="flux-schnell", prompt="a cat")
        assert r.adapters == []
        assert r.adapter_weights is None

    def test_adapters_only(self) -> None:
        r = DiffusionRequest(
            model_name="flux-schnell",
            prompt="a cat",
            adapters=["sketch"],
        )
        assert r.adapters == ["sketch"]
        assert r.adapter_weights is None

    def test_adapters_and_weights(self) -> None:
        r = DiffusionRequest(
            model_name="flux-schnell",
            prompt="a cat",
            adapters=["sketch", "watercolor"],
            adapter_weights=[1.0, 0.5],
        )
        assert r.adapters == ["sketch", "watercolor"]
        assert r.adapter_weights == [1.0, 0.5]

    def test_weights_without_adapters_rejected(self) -> None:
        with pytest.raises(ValidationError, match="adapters is empty"):
            DiffusionRequest(
                model_name="flux-schnell",
                prompt="a cat",
                adapter_weights=[1.0],
            )

    def test_weights_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="length"):
            DiffusionRequest(
                model_name="flux-schnell",
                prompt="a cat",
                adapters=["sketch", "watercolor"],
                adapter_weights=[1.0],
            )


class TestMultimodalGenerationRequestAdapters:
    def test_defaults(self) -> None:
        r = MultimodalGenerationRequest(
            model_name="sdxl-img2img",
            prompt="a cat",
            image_b64=_PNG_B64,
        )
        assert r.adapters == []
        assert r.adapter_weights is None

    def test_adapters_and_weights(self) -> None:
        r = MultimodalGenerationRequest(
            model_name="sdxl-img2img",
            prompt="a cat",
            image_b64=_PNG_B64,
            adapters=["pixar"],
            adapter_weights=[0.8],
        )
        assert r.adapters == ["pixar"]
        assert r.adapter_weights == [0.8]

    def test_weights_without_adapters_rejected(self) -> None:
        with pytest.raises(ValidationError, match="adapters is empty"):
            MultimodalGenerationRequest(
                model_name="sdxl-img2img",
                prompt="a cat",
                image_b64=_PNG_B64,
                adapter_weights=[0.5],
            )

    def test_weights_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError, match="length"):
            MultimodalGenerationRequest(
                model_name="sdxl-img2img",
                prompt="a cat",
                image_b64=_PNG_B64,
                adapters=["a", "b", "c"],
                adapter_weights=[1.0, 1.0],
            )
