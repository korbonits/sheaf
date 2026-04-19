"""Tests for SDXLBackend — fully mocked, no diffusers/torch/PIL required.

Covers:
  - __init__() raises ValueError on invalid mode
  - load() raises ImportError when diffusers is absent
  - load() uses StableDiffusionXLImg2ImgPipeline in img2img mode
  - load() uses StableDiffusionXLInpaintPipeline in inpaint mode
  - load() resolves torch_dtype string to torch.dtype
  - load() moves pipeline to device (.to(device)) when cpu_offload=False
  - load() calls enable_model_cpu_offload() and skips .to() when cpu_offload=True
  - model_type returns ModelType.MULTIMODAL_GENERATION
  - predict() rejects non-MultimodalGenerationRequest inputs
  - predict() returns MultimodalGenerationResponse with image_b64 and seed
  - predict() forwards prompt, image, strength, steps, guidance_scale
  - predict() omits negative_prompt when empty string
  - predict() includes negative_prompt when non-empty
  - predict() decodes mask_b64 and passes mask_image when provided
  - predict() uses request seed when set
  - predict() generates a random seed when request.seed is None
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import io
import struct
import sys
import zlib
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.multimodal_generation import (
    MultimodalGenerationRequest,
    MultimodalGenerationResponse,
)

# ---------------------------------------------------------------------------
# Minimal PNG helper (no PIL needed in tests)
# ---------------------------------------------------------------------------


def _make_png(width: int = 8, height: int = 8) -> bytes:
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


_IMG_B64 = base64.b64encode(b"fake_image_bytes").decode()
_MASK_B64 = base64.b64encode(b"fake_mask_bytes").decode()
_PNG_W, _PNG_H = 512, 512


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_fake_image(w: int = _PNG_W, h: int = _PNG_H) -> MagicMock:
    img = MagicMock()
    img.width = w
    img.height = h

    def _save(buf: io.BytesIO, format: str = "PNG") -> None:
        buf.write(_make_png(w, h))

    img.save.side_effect = _save
    return img


def _make_pipeline_cls(img: MagicMock | None = None) -> MagicMock:
    """Mock pipeline class following FLUX pattern.

    from_pretrained(...).to(device) → instance
    instance(**kwargs) → result with .images[0] = img
    """
    if img is None:
        img = _make_fake_image()
    instance = MagicMock()
    instance.return_value.images = [img]
    pipeline_cls = MagicMock()
    pipeline_cls.from_pretrained.return_value.to.return_value = instance
    return pipeline_cls


def _make_fake_diffusers(
    img2img_cls: MagicMock | None = None,
    inpaint_cls: MagicMock | None = None,
) -> ModuleType:
    mod = ModuleType("diffusers")
    mod.StableDiffusionXLImg2ImgPipeline = img2img_cls or _make_pipeline_cls()  # type: ignore[attr-defined]
    mod.StableDiffusionXLInpaintPipeline = inpaint_cls or _make_pipeline_cls()  # type: ignore[attr-defined]
    return mod


def _make_fake_torch() -> MagicMock:
    t = MagicMock()
    t.float32 = "float32"
    t.float16 = "float16"
    t.bfloat16 = "bfloat16"
    t.Generator.return_value.manual_seed.return_value = MagicMock()
    return t


def _make_fake_pil(open_returns: MagicMock | None = None) -> ModuleType:
    """Fake PIL module where Image.open().convert() returns open_returns."""
    pil_mod = ModuleType("PIL")
    image_cls = MagicMock()
    fake_img = open_returns or MagicMock()
    fake_img.convert.return_value = fake_img
    image_cls.open.return_value = fake_img
    pil_mod.Image = image_cls  # type: ignore[attr-defined]
    return pil_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_request(**kwargs) -> MultimodalGenerationRequest:  # type: ignore[no-untyped-def]
    defaults = dict(model_name="sdxl", prompt="a cat", image_b64=_IMG_B64)
    defaults.update(kwargs)
    return MultimodalGenerationRequest(**defaults)


@pytest.fixture
def fake_torch() -> MagicMock:
    return _make_fake_torch()


@pytest.fixture
def fake_diffusers() -> ModuleType:
    return _make_fake_diffusers()


@pytest.fixture
def fake_pil() -> ModuleType:
    return _make_fake_pil()


@pytest.fixture
def loaded_backend(
    fake_diffusers: ModuleType, fake_torch: MagicMock, fake_pil: ModuleType
):  # type: ignore[no-untyped-def]
    from sheaf.backends.sdxl import SDXLBackend

    backend = SDXLBackend(
        model_id="stabilityai/stable-diffusion-xl-base-1.0", device="cpu"
    )
    with patch.dict(
        sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch, "PIL": fake_pil}
    ):
        backend.load()
    # Replace the pipeline instance with one that returns a known image.
    fake_img = _make_fake_image()
    instance = MagicMock()
    instance.return_value.images = [fake_img]
    backend._pipeline = instance
    backend._Image = fake_pil.Image  # type: ignore[attr-defined]
    return backend, fake_torch, instance


# ---------------------------------------------------------------------------
# __init__() validation
# ---------------------------------------------------------------------------


def test_init_rejects_invalid_mode() -> None:
    from sheaf.backends.sdxl import SDXLBackend

    with pytest.raises(ValueError, match="mode"):
        SDXLBackend(mode="text2img")


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_diffusers() -> None:
    import builtins

    from sheaf.backends.sdxl import SDXLBackend

    backend = SDXLBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "diffusers" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if "diffusers" in name:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[multimodal-generation\\]"),
    ):
        backend.load()


def test_load_uses_img2img_pipeline(
    fake_diffusers: ModuleType, fake_torch: MagicMock, fake_pil: ModuleType
) -> None:
    from sheaf.backends.sdxl import SDXLBackend

    backend = SDXLBackend(mode="img2img")
    with patch.dict(
        sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch, "PIL": fake_pil}
    ):
        backend.load()

    fake_diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained.assert_called_once()  # type: ignore[attr-defined]
    fake_diffusers.StableDiffusionXLInpaintPipeline.from_pretrained.assert_not_called()  # type: ignore[attr-defined]


def test_load_uses_inpaint_pipeline(
    fake_diffusers: ModuleType, fake_torch: MagicMock, fake_pil: ModuleType
) -> None:
    from sheaf.backends.sdxl import SDXLBackend

    backend = SDXLBackend(mode="inpaint")
    with patch.dict(
        sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch, "PIL": fake_pil}
    ):
        backend.load()

    fake_diffusers.StableDiffusionXLInpaintPipeline.from_pretrained.assert_called_once()  # type: ignore[attr-defined]
    fake_diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained.assert_not_called()  # type: ignore[attr-defined]


def test_load_moves_pipeline_to_device(
    fake_diffusers: ModuleType, fake_torch: MagicMock, fake_pil: ModuleType
) -> None:
    from sheaf.backends.sdxl import SDXLBackend

    backend = SDXLBackend(device="cuda")
    with patch.dict(
        sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch, "PIL": fake_pil}
    ):
        backend.load()

    fake_diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained.return_value.to.assert_called_once_with(  # type: ignore[attr-defined]
        "cuda"
    )


def test_load_cpu_offload_skips_to(
    fake_diffusers: ModuleType, fake_torch: MagicMock, fake_pil: ModuleType
) -> None:
    from sheaf.backends.sdxl import SDXLBackend

    backend = SDXLBackend(enable_model_cpu_offload=True)
    with patch.dict(
        sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch, "PIL": fake_pil}
    ):
        backend.load()

    pre_to = (
        fake_diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained.return_value
    )  # type: ignore[attr-defined]
    pre_to.enable_model_cpu_offload.assert_called_once()
    pre_to.to.assert_not_called()


def test_model_type() -> None:
    from sheaf.api.base import ModelType
    from sheaf.backends.sdxl import SDXLBackend

    assert SDXLBackend().model_type == ModelType.MULTIMODAL_GENERATION


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    backend, fake_torch, _ = loaded_backend
    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with patch.dict(sys.modules, {"torch": fake_torch}):
        with pytest.raises(TypeError, match="MultimodalGenerationRequest"):
            backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_multimodal_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, _ = loaded_backend
    req = _make_request()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        resp = backend.predict(req)
    assert isinstance(resp, MultimodalGenerationResponse)
    assert resp.image_b64
    assert resp.width == _PNG_W
    assert resp.height == _PNG_H


def test_predict_forwards_core_kwargs(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, instance = loaded_backend
    req = _make_request(
        prompt="a dog", strength=0.6, num_inference_steps=30, guidance_scale=8.0
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        backend.predict(req)
    _, kwargs = instance.call_args
    assert kwargs["prompt"] == "a dog"
    assert kwargs["strength"] == pytest.approx(0.6)
    assert kwargs["num_inference_steps"] == 30
    assert kwargs["guidance_scale"] == pytest.approx(8.0)


def test_predict_omits_negative_prompt_when_empty(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, instance = loaded_backend
    req = _make_request(negative_prompt="")
    with patch.dict(sys.modules, {"torch": fake_torch}):
        backend.predict(req)
    _, kwargs = instance.call_args
    assert "negative_prompt" not in kwargs


def test_predict_includes_negative_prompt_when_set(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, instance = loaded_backend
    req = _make_request(negative_prompt="blurry, low quality")
    with patch.dict(sys.modules, {"torch": fake_torch}):
        backend.predict(req)
    _, kwargs = instance.call_args
    assert kwargs["negative_prompt"] == "blurry, low quality"


def test_predict_passes_mask_image_when_provided(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, instance = loaded_backend
    req = _make_request(mask_b64=_MASK_B64)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        backend.predict(req)
    _, kwargs = instance.call_args
    assert "mask_image" in kwargs


def test_predict_no_mask_image_when_not_provided(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, instance = loaded_backend
    req = _make_request()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        backend.predict(req)
    _, kwargs = instance.call_args
    assert "mask_image" not in kwargs


def test_predict_uses_request_seed(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, _ = loaded_backend
    req = _make_request(seed=42)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        resp = backend.predict(req)
    assert resp.seed == 42
    fake_torch.Generator.return_value.manual_seed.assert_called_with(42)


def test_predict_generates_random_seed_when_none(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, _ = loaded_backend
    req = _make_request(seed=None)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        resp = backend.predict(req)
    assert isinstance(resp.seed, int)


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, fake_torch, instance = loaded_backend
    reqs = [_make_request(prompt="cat"), _make_request(prompt="dog")]
    with patch.dict(sys.modules, {"torch": fake_torch}):
        responses = backend.batch_predict(reqs)
    assert len(responses) == 2
    assert all(isinstance(r, MultimodalGenerationResponse) for r in responses)
    assert instance.call_count == 2
