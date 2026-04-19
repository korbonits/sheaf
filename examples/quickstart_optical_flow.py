"""RAFT optical flow quickstart.

Requirements:
    pip install "sheaf-serve[optical-flow]"
    pip install Pillow  # for generating example frames

Usage:
    python examples/quickstart_optical_flow.py

Demonstrates:
  - RAFT-large dense optical flow between two frames
  - RAFT-small (faster, smaller model)
  - Flow magnitude and direction statistics
  - Synthetic translating object: constant non-zero flow expected
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

from sheaf.api.optical_flow import OpticalFlowRequest
from sheaf.backends.raft import RAFTBackend

# ---------------------------------------------------------------------------
# Minimal PNG builder
# ---------------------------------------------------------------------------


def _png(pixels: list[list[tuple[int, int, int]]], width: int, height: int) -> bytes:
    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    rows = b""
    for row in pixels:
        rows += b"\x00" + b"".join(bytes(p) for p in row)
    compressed = zlib.compress(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _checkerboard_png(offset_x: int = 0, width: int = 256, height: int = 256) -> bytes:
    """Checkerboard pattern with a horizontal offset to simulate motion."""
    cell = 32
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            if ((x + offset_x) // cell + y // cell) % 2 == 0:
                row.append((200, 200, 200))
            else:
                row.append((50, 50, 50))
        pixels.append(row)
    return _png(pixels, width, height)


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


# ---------------------------------------------------------------------------
# Synthetic frames: checkerboard shifted 16 px to the right
# ---------------------------------------------------------------------------

W, H = 256, 256
SHIFT = 16  # pixels

print(f"Building synthetic frames ({W}×{H}, {SHIFT}px rightward shift)...")
frame1_b64 = _b64(_checkerboard_png(offset_x=0, width=W, height=H))
frame2_b64 = _b64(_checkerboard_png(offset_x=SHIFT, width=W, height=H))
print()

# ---------------------------------------------------------------------------
# RAFT-large
# ---------------------------------------------------------------------------

print("--- RAFT-large (downloads ~20 MB on first run) ---")
print("Loading model...")

backend = RAFTBackend(model_name="raft_large", device="cpu")
backend.load()
print("Model loaded.")

req = OpticalFlowRequest(
    model_name="raft",
    frame1_b64=frame1_b64,
    frame2_b64=frame2_b64,
)
resp = backend.predict(req)

flow = np.frombuffer(base64.b64decode(resp.flow_b64), dtype=np.float32).reshape(
    resp.height, resp.width, 2
)
dx = flow[..., 0]
dy = flow[..., 1]
mag = np.sqrt(dx**2 + dy**2)

print(f"\nFlow field shape : {flow.shape}  (H, W, 2)")
print(
    f"dx  mean={dx.mean():.2f}  std={dx.std():.2f}  "
    f"min={dx.min():.2f}  max={dx.max():.2f}"
)
print(
    f"dy  mean={dy.mean():.2f}  std={dy.std():.2f}  "
    f"min={dy.min():.2f}  max={dy.max():.2f}"
)
print(f"mag mean={mag.mean():.2f}  max={mag.max():.2f}")
print(f"Expected: dx ≈ -{SHIFT:.1f} (frame2 shifted right → negative flow), dy ≈ 0")

# ---------------------------------------------------------------------------
# RAFT-small
# ---------------------------------------------------------------------------

print("\n--- RAFT-small (faster, ~5 MB) ---")
print("Loading model...")

backend_small = RAFTBackend(model_name="raft_small", device="cpu")
backend_small.load()
print("Model loaded.")

resp_s = backend_small.predict(req)
flow_s = np.frombuffer(base64.b64decode(resp_s.flow_b64), dtype=np.float32).reshape(
    resp_s.height, resp_s.width, 2
)
dx_s = flow_s[..., 0]
dy_s = flow_s[..., 1]

print(f"\nFlow field shape : {flow_s.shape}")
print(f"dx  mean={dx_s.mean():.2f}  std={dx_s.std():.2f}")
print(f"dy  mean={dy_s.mean():.2f}  std={dy_s.std():.2f}")

print("\nDone.")
