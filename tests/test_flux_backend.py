"""Tests for FluxBackend — mocked, no diffusers/GPU required."""

from __future__ import annotations

import base64
import builtins
import io
import struct
import sys
import zlib
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.diffusion import DiffusionRequest, DiffusionResponse

# ---------------------------------------------------------------------------
# Minimal valid PNG builder (no PIL needed in tests)
# ---------------------------------------------------------------------------


def _make_png(width: int = 8, height: int = 8) -> bytes:
    """Return a minimal valid RGB PNG of the given dimensions."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    raw = b"\x00" + b"\xff\xff\xff" * width
    compressed = zlib.compress(raw * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_pipeline(width: int = 1024, height: int = 1024) -> MagicMock:
    """Return a MagicMock FluxPipeline class that returns a PIL-like image.

    load() calls: FluxPipeline.from_pretrained(...).to(device)
    predict() calls: self._pipeline(prompt=..., ...)  → result.images[0]
    """
    fake_img = MagicMock()
    fake_img.width = width
    fake_img.height = height

    def _save(buf: io.BytesIO, format: str = "PNG") -> None:
        buf.write(_make_png(width, height))

    fake_img.save.side_effect = _save

    # The instance returned after .from_pretrained(...).to(device)
    instance = MagicMock()
    instance.return_value.images = [fake_img]  # instance(prompt=...) → .images[0]

    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value.to.return_value = instance
    return pipeline_cls


def _make_fake_diffusers(pipeline: MagicMock) -> MagicMock:
    mod = MagicMock()
    mod.FluxPipeline = pipeline
    return mod


def _make_fake_torch() -> MagicMock:
    torch = MagicMock()
    torch.float32 = "float32"
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    torch.Generator.return_value.manual_seed.return_value = MagicMock()
    return torch


def _loaded_backend(model_id: str = "black-forest-labs/FLUX.1-schnell") -> tuple:
    """Return (backend, fake_pipeline) with load() already called."""
    from sheaf.backends.flux import FluxBackend

    fake_pipeline = _make_fake_pipeline()
    fake_diffusers = _make_fake_diffusers(fake_pipeline)
    fake_torch = _make_fake_torch()

    with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
        backend = FluxBackend(model_id=model_id, device="cpu")
        backend.load()

    # Attach fakes so tests can inspect calls
    backend._fake_pipeline = fake_pipeline
    backend._fake_torch = fake_torch
    return backend, fake_pipeline, fake_torch


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


class TestDiffusionRequest:
    def test_defaults(self) -> None:
        req = DiffusionRequest(model_name="flux", prompt="a red apple")
        assert req.height == 1024
        assert req.width == 1024
        assert req.num_inference_steps == 4
        assert req.guidance_scale == 0.0
        assert req.seed is None
        assert req.negative_prompt == ""

    def test_custom_dimensions(self) -> None:
        req = DiffusionRequest(model_name="flux", prompt="test", height=512, width=768)
        assert req.height == 512
        assert req.width == 768

    def test_rejects_empty_prompt(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="prompt"):
            DiffusionRequest(model_name="flux", prompt="   ")

    def test_rejects_non_multiple_of_8(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DiffusionRequest(model_name="flux", prompt="test", height=100)

    def test_rejects_zero_steps(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DiffusionRequest(model_name="flux", prompt="test", num_inference_steps=0)


# ---------------------------------------------------------------------------
# FluxBackend unit tests
# ---------------------------------------------------------------------------


class TestFluxBackend:
    def test_load_calls_from_pretrained(self) -> None:
        from sheaf.backends.flux import FluxBackend

        fake_pipeline = _make_fake_pipeline()
        fake_diffusers = _make_fake_diffusers(fake_pipeline)
        fake_torch = _make_fake_torch()

        with patch.dict(
            sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}
        ):
            backend = FluxBackend(
                model_id="black-forest-labs/FLUX.1-schnell", device="cpu"
            )
            backend.load()

        fake_pipeline.from_pretrained.assert_called_once_with(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype="bfloat16",
        )

    def test_load_raises_without_diffusers(self) -> None:
        _real_import = builtins.__import__

        def _block_diffusers(name, *args, **kwargs):
            if name == "diffusers":
                raise ImportError("No module named 'diffusers'")
            return _real_import(name, *args, **kwargs)

        from sheaf.backends.flux import FluxBackend

        with patch("builtins.__import__", side_effect=_block_diffusers):
            backend = FluxBackend()
            with pytest.raises(ImportError, match="diffusers"):
                backend.load()

    def test_predict_returns_diffusion_response(self) -> None:
        backend, _, _ = _loaded_backend()
        req = DiffusionRequest(
            model_name="flux-schnell", prompt="a cat on a moon", seed=42
        )
        fake_torch = _make_fake_torch()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)

        assert isinstance(resp, DiffusionResponse)
        assert resp.model_name == "flux-schnell"
        assert resp.seed == 42

    def test_predict_image_b64_is_valid_png(self) -> None:
        backend, _, _ = _loaded_backend()
        req = DiffusionRequest(model_name="flux", prompt="landscape", seed=1)
        fake_torch = _make_fake_torch()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)

        raw = base64.b64decode(resp.image_b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    def test_predict_uses_request_seed(self) -> None:
        backend, fake_pipeline, fake_torch = _loaded_backend()
        req = DiffusionRequest(model_name="flux", prompt="test", seed=1234)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)
        assert resp.seed == 1234

    def test_predict_generates_seed_when_none(self) -> None:
        backend, _, fake_torch = _loaded_backend()
        req = DiffusionRequest(model_name="flux", prompt="test", seed=None)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)
        assert isinstance(resp.seed, int)

    def test_predict_passes_dimensions_to_pipeline(self) -> None:
        backend, fake_pipeline, fake_torch = _loaded_backend()
        req = DiffusionRequest(
            model_name="flux", prompt="test", height=512, width=768, seed=0
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            backend.predict(req)
        instance = fake_pipeline.from_pretrained.return_value.to.return_value
        _, kwargs = instance.call_args
        assert kwargs["height"] == 512
        assert kwargs["width"] == 768

    def test_predict_passes_steps_and_guidance(self) -> None:
        backend, fake_pipeline, fake_torch = _loaded_backend()
        req = DiffusionRequest(
            model_name="flux",
            prompt="test",
            num_inference_steps=20,
            guidance_scale=3.5,
            seed=0,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            backend.predict(req)
        instance = fake_pipeline.from_pretrained.return_value.to.return_value
        _, kwargs = instance.call_args
        assert kwargs["num_inference_steps"] == 20
        assert kwargs["guidance_scale"] == 3.5

    def test_predict_raises_before_load(self) -> None:
        from sheaf.backends.flux import FluxBackend

        backend = FluxBackend()
        req = DiffusionRequest(model_name="flux", prompt="test")
        with pytest.raises(RuntimeError, match="load()"):
            backend.predict(req)

    def test_batch_predict_runs_each_request(self) -> None:
        backend, _, fake_torch = _loaded_backend()
        reqs = [
            DiffusionRequest(model_name="flux", prompt=f"image {i}", seed=i)
            for i in range(3)
        ]
        with patch.dict(sys.modules, {"torch": fake_torch}):
            results = backend.batch_predict(reqs)
        assert len(results) == 3
        assert all(isinstance(r, DiffusionResponse) for r in results)

    def test_cpu_offload_called_when_enabled(self) -> None:
        from sheaf.backends.flux import FluxBackend

        fake_pipeline = _make_fake_pipeline()
        fake_diffusers = _make_fake_diffusers(fake_pipeline)
        fake_torch = _make_fake_torch()
        instance = fake_pipeline.from_pretrained.return_value

        with patch.dict(
            sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}
        ):
            backend = FluxBackend(enable_model_cpu_offload=True)
            backend.load()

        instance.enable_model_cpu_offload.assert_called_once()
        instance.to.assert_not_called()

    def test_model_type_is_diffusion(self) -> None:
        from sheaf.api.base import ModelType
        from sheaf.backends.flux import FluxBackend

        assert FluxBackend().model_type == ModelType.DIFFUSION
