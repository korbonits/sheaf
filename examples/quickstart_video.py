"""VideoMAE video understanding quickstart: embeddings and action classification.

Requirements:
    pip install "sheaf-serve[video]"

    VideoMAE-base downloads ~350 MB on first run.
    Runs on CPU; a GPU is recommended for real workloads.

Usage:
    python examples/quickstart_video.py

Demonstrates:
  - Synthetic video clips (colored frames — no video file required)
  - VideoMAE-base 768-dim embeddings with CLS and mean pooling
  - Cosine similarity between two different "motion" clips
  - TimeSformer action classification (top-5 softmax over Kinetics-400)
"""

from __future__ import annotations

import base64
import math
import struct
import zlib

from sheaf.api.video import VideoRequest
from sheaf.backends.videomae import VideoMAEBackend

# ---------------------------------------------------------------------------
# Minimal PNG builder — no PIL / Pillow required in this file
# ---------------------------------------------------------------------------


def _solid_png(r: int, g: int, b: int, width: int = 224, height: int = 224) -> bytes:
    """Return a valid PNG with a solid colour fill (no external dependencies)."""

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    # Each scan line: filter byte 0x00 + RGB triplets
    row = b"\x00" + bytes([r, g, b]) * width
    compressed = zlib.compress(row * height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _frame_b64(r: int, g: int, b: int) -> str:
    return base64.b64encode(_solid_png(r, g, b)).decode()


def _gradient_clip(
    n_frames: int = 16, base_color: tuple[int, int, int] = (200, 50, 50)
) -> list[str]:
    """Return n_frames base64-encoded PNGs that shift brightness over time."""
    r0, g0, b0 = base_color
    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frames.append(
            _frame_b64(
                min(255, int(r0 + t * 40)),
                min(255, int(g0 + t * 40)),
                min(255, int(b0 + t * 40)),
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Build synthetic clips
# ---------------------------------------------------------------------------

print("Building synthetic video clips (16 frames each)...")

# "Red motion" clip: warm red gradient
clip_red = _gradient_clip(n_frames=16, base_color=(200, 50, 50))

# "Blue motion" clip: cool blue gradient
clip_blue = _gradient_clip(n_frames=16, base_color=(50, 50, 200))

# "Red similar" clip: another warm red — should be closer to clip_red
clip_red2 = _gradient_clip(n_frames=16, base_color=(210, 60, 40))

n_chars = len(clip_red[0])
print(f"  Each clip: {len(clip_red)} frames, {n_chars} chars/frame (base64 PNG)\n")

# ---------------------------------------------------------------------------
# VideoMAE embedding — CLS pooling
# ---------------------------------------------------------------------------

print("--- VideoMAE-base (MCG-NJU/videomae-base, 768-dim, CLS pooling) ---")
print("Loading model (downloads ~350 MB on first run)...")

backend_cls = VideoMAEBackend(
    model_name="MCG-NJU/videomae-base",
    task="embedding",
    device="cpu",
    pooling="cls",
)
backend_cls.load()
print("Model loaded.")

req_red = VideoRequest(model_name="videomae", frames_b64=clip_red, normalize=True)
req_blue = VideoRequest(model_name="videomae", frames_b64=clip_blue, normalize=True)
req_red2 = VideoRequest(model_name="videomae", frames_b64=clip_red2, normalize=True)

resp_red = backend_cls.predict(req_red)
resp_blue = backend_cls.predict(req_blue)
resp_red2 = backend_cls.predict(req_red2)

print(f"\nEmbedding dim : {resp_red.dim}")
print("\nCosine similarities (CLS pooling, L2-normalized):")
print(f"  red  vs blue  : {cosine(resp_red.embedding, resp_blue.embedding):.4f}")
print(f"  red  vs red2  : {cosine(resp_red.embedding, resp_red2.embedding):.4f}")
print(f"  blue vs red2  : {cosine(resp_blue.embedding, resp_red2.embedding):.4f}")
print("  (red vs red2 should be highest)")

# ---------------------------------------------------------------------------
# VideoMAE embedding — mean pooling
# ---------------------------------------------------------------------------

print("\n--- VideoMAE-base (mean pooling over patch tokens) ---")

backend_mean = VideoMAEBackend(
    model_name="MCG-NJU/videomae-base",
    task="embedding",
    device="cpu",
    pooling="mean",
)
backend_mean.load()

resp_red_m = backend_mean.predict(req_red)
resp_blue_m = backend_mean.predict(req_blue)
resp_red2_m = backend_mean.predict(req_red2)

print(f"\nEmbedding dim : {resp_red_m.dim}")
print("\nCosine similarities (mean pooling, L2-normalized):")
print(f"  red  vs blue  : {cosine(resp_red_m.embedding, resp_blue_m.embedding):.4f}")
print(f"  red  vs red2  : {cosine(resp_red_m.embedding, resp_red2_m.embedding):.4f}")
print(f"  blue vs red2  : {cosine(resp_blue_m.embedding, resp_red2_m.embedding):.4f}")

# ---------------------------------------------------------------------------
# VideoMAE fine-tuned on Kinetics-400 — action classification
# ---------------------------------------------------------------------------

print("\n--- VideoMAE-base fine-tuned Kinetics-400 (classification, top-5) ---")
print("Loading classification model (downloads ~350 MB on first run)...")

backend_clf = VideoMAEBackend(
    model_name="MCG-NJU/videomae-base-finetuned-kinetics",
    task="classification",
    device="cpu",
)
backend_clf.load()
print("Model loaded.")

req_clf = VideoRequest(
    model_name="videomae-kinetics",
    frames_b64=clip_red,
    task="classification",
)
resp_clf = backend_clf.predict(req_clf)

print("\nTop-5 predicted actions:")
for label, score in zip(resp_clf.labels, resp_clf.scores):
    bar = "#" * int(score * 40)
    print(f"  {label:30s}  {score:.4f}  {bar}")

# ---------------------------------------------------------------------------
# TimeSformer — 8-frame model (same classification API)
# ---------------------------------------------------------------------------

print("\n--- TimeSformer-base fine-tuned K400 (classification, 8 frames) ---")
print("Loading model (downloads ~400 MB on first run)...")

clip_8 = clip_red[:8]  # TimeSformer expects 8 frames

backend_tsf = VideoMAEBackend(
    model_name="facebook/timesformer-base-finetuned-k400",
    task="classification",
    device="cpu",
)
backend_tsf.load()
print("Model loaded.")

req_tsf = VideoRequest(
    model_name="timesformer",
    frames_b64=clip_8,
    task="classification",
)
resp_tsf = backend_tsf.predict(req_tsf)

print("\nTop-5 predicted actions (TimeSformer):")
for label, score in zip(resp_tsf.labels, resp_tsf.scores):
    bar = "#" * int(score * 40)
    print(f"  {label:30s}  {score:.4f}  {bar}")

print("\nDone.")
