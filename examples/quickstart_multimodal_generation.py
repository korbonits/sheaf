"""SDXL multimodal generation quickstart (img2img and inpainting).

Requirements:
    pip install "sheaf-serve[multimodal-generation]"
    pip install Pillow  # for loading and saving images

Usage:
    python examples/quickstart_multimodal_generation.py

Demonstrates:
  - SDXL img2img: style-transfer a solid-colour source image
  - SDXL inpainting: re-generate a masked region within the source image
  - Seed reproducibility: same seed → same output
  - Strength control: 0.5 (preserve structure) vs 0.9 (free generation)

Note: SDXL downloads ~7 GB on first run and requires a GPU for practical
      inference speed.  On CPU each call takes several minutes.
      For a lighter demo, pass model_id="stabilityai/sdxl-turbo" (4-step,
      Apache 2.0, ~7 GB) and set num_inference_steps=4, guidance_scale=0.0.
"""

from __future__ import annotations

import base64
import struct
import zlib

from sheaf.api.multimodal_generation import MultimodalGenerationRequest
from sheaf.backends.sdxl import SDXLBackend

# ---------------------------------------------------------------------------
# Minimal PNG builder — no Pillow dependency for frame generation
# ---------------------------------------------------------------------------


def _solid_png(r: int, g: int, b: int, width: int = 512, height: int = 512) -> bytes:
    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    row = b"\x00" + bytes([r, g, b]) * width
    compressed = zlib.compress(row * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _mask_png(width: int = 512, height: int = 512) -> bytes:
    """White centre square on black background — inpaint the centre region."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    cx, cy = width // 2, height // 2
    hw, hh = width // 4, height // 4
    rows = b""
    for y in range(height):
        row = b"\x00"
        for x in range(width):
            if cx - hw <= x <= cx + hw and cy - hh <= y <= cy + hh:
                row += b"\xff\xff\xff"
            else:
                row += b"\x00\x00\x00"
        rows += row
    compressed = zlib.compress(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def save_png(image_b64: str, path: str, label: str) -> None:
    png_bytes = base64.b64decode(image_b64)
    with open(path, "wb") as f:
        f.write(png_bytes)
    print(f"  Saved {path}  ({len(png_bytes):,} bytes)  — {label}")


# ---------------------------------------------------------------------------
# Source images
# ---------------------------------------------------------------------------

W, H = 512, 512
source_b64 = _b64(_solid_png(100, 149, 237, width=W, height=H))  # cornflower blue
mask_b64 = _b64(_mask_png(width=W, height=H))

# ---------------------------------------------------------------------------
# SDXL img2img — style transfer
# ---------------------------------------------------------------------------

print("--- SDXL img2img (downloads ~7 GB on first run; GPU recommended) ---")
print("Loading model...")

backend_i2i = SDXLBackend(
    model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
    mode="img2img",
    device="cuda",
    torch_dtype="bfloat16",
)
backend_i2i.load()
print("Model loaded.\n")

req = MultimodalGenerationRequest(
    model_name="sdxl-i2i",
    prompt="a serene mountain lake at golden hour, photorealistic, 8k",
    image_b64=source_b64,
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=7.5,
    negative_prompt="blurry, low quality, watermark",
    seed=42,
)
resp = backend_i2i.predict(req)
save_png(resp.image_b64, "/tmp/sdxl_img2img.png", "img2img (strength=0.75, seed=42)")

# Reproducibility: same seed → identical output
resp2 = backend_i2i.predict(req)
same_pixels = resp.image_b64 == resp2.image_b64
print(f"  Seed reproducibility: {same_pixels}  (same_pixels)")

# Strength comparison
for strength in [0.5, 0.9]:
    req_s = MultimodalGenerationRequest(
        model_name="sdxl-i2i",
        prompt="a snowy alpine meadow, painterly",
        image_b64=source_b64,
        strength=strength,
        num_inference_steps=20,
        seed=7,
    )
    resp_s = backend_i2i.predict(req_s)
    save_png(
        resp_s.image_b64,
        f"/tmp/sdxl_strength_{str(strength).replace('.', '_')}.png",
        f"strength={strength}",
    )

# ---------------------------------------------------------------------------
# SDXL inpainting
# ---------------------------------------------------------------------------

print("\n--- SDXL inpainting ---")
print("Loading inpaint model...")

backend_inp = SDXLBackend(
    model_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    mode="inpaint",
    device="cuda",
    torch_dtype="bfloat16",
)
backend_inp.load()
print("Model loaded.\n")

req_inp = MultimodalGenerationRequest(
    model_name="sdxl-inpaint",
    prompt="a glowing crystal sphere, fantasy art, dramatic lighting",
    image_b64=source_b64,
    mask_b64=mask_b64,
    strength=0.99,
    num_inference_steps=30,
    guidance_scale=8.0,
    seed=123,
)
resp_inp = backend_inp.predict(req_inp)
save_png(resp_inp.image_b64, "/tmp/sdxl_inpaint.png", "inpainting centre region")

print(f"\n  Output size: {resp_inp.width}×{resp_inp.height}  seed={resp_inp.seed}")

print("\nDone.")
