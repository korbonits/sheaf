"""ViTPose human pose estimation quickstart.

Requirements:
    pip install "sheaf-serve[pose]"
    pip install Pillow  # for generating the example image

Usage:
    python examples/quickstart_pose.py

Demonstrates:
  - ViTPose-base (COCO 17-keypoint skeleton)
  - Full-image inference (no explicit bounding boxes)
  - Explicit bounding box crops for multi-person images
  - Per-keypoint confidence filtering
  - Keypoint name lookup from model config
"""

from __future__ import annotations

import base64
import struct
import zlib

from sheaf.api.pose import PoseRequest
from sheaf.backends.vitpose import ViTPoseBackend

# ---------------------------------------------------------------------------
# Minimal PNG builder — no Pillow required for basic examples
# ---------------------------------------------------------------------------


def _solid_png(r: int, g: int, b: int, width: int = 192, height: int = 256) -> bytes:
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


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


# ---------------------------------------------------------------------------
# Load backend
# ---------------------------------------------------------------------------

print("Loading ViTPose-base (downloads ~330 MB on first run)...")

backend = ViTPoseBackend(
    model_name="usyd-community/vitpose-base-simple",
    device="cpu",
)
backend.load()
print("Model loaded.\n")
print(f"Keypoints ({len(backend._keypoint_names())} total):")
for i, name in enumerate(backend._keypoint_names()):
    print(f"  {i:2d}  {name}")
print()

# ---------------------------------------------------------------------------
# Single-person inference — full image as one crop
# ---------------------------------------------------------------------------

print("--- Single-person inference (full-image crop, 192×256 px) ---")

image_b64 = _b64(_solid_png(180, 140, 100, width=192, height=256))
req = PoseRequest(model_name="vitpose", image_b64=image_b64)
resp = backend.predict(req)

print(f"  Detected {len(resp.poses)} person(s)")
print(f"  Image size: {resp.width}×{resp.height}")
print(f"  Keypoints per person: {len(resp.poses[0]) if resp.poses else 0}")

if resp.poses:
    print("\n  Keypoint positions (x, y, confidence):")
    for kp, name in zip(resp.poses[0], resp.keypoint_names):
        x, y, score = kp
        bar = "#" * int(score * 20)
        print(f"    {name:15s}  ({x:6.1f}, {y:6.1f})  {score:.3f}  {bar}")

# ---------------------------------------------------------------------------
# Explicit bounding boxes — simulate two-person crop
# ---------------------------------------------------------------------------

print("\n--- Two-person inference with explicit bounding boxes ---")

wide_image_b64 = _b64(_solid_png(100, 160, 200, width=640, height=480))
bboxes = [
    [50.0, 20.0, 250.0, 460.0],  # left person
    [380.0, 20.0, 590.0, 460.0],  # right person
]
req2 = PoseRequest(
    model_name="vitpose",
    image_b64=wide_image_b64,
    bboxes=bboxes,
)
resp2 = backend.predict(req2)

print(f"  Detected {len(resp2.poses)} person(s)")
for i, person_kps in enumerate(resp2.poses):
    high_conf = [
        (kp, name) for kp, name in zip(person_kps, resp2.keypoint_names) if kp[2] >= 0.3
    ]
    print(f"  Person {i}: {len(high_conf)} keypoints with confidence ≥ 0.3")

print("\nDone.")
