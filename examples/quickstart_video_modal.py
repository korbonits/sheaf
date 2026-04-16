"""VideoMAE video understanding on Modal — embeddings and classification on a T4 GPU.

Runs VideoMAE-base on a T4 via Modal cloud compute.

Prerequisites:
    pip install modal
    modal setup              # authenticate once

Run:
    modal run examples/quickstart_video_modal.py

The model is cached inside the Modal volume after the first run (~350 MB).
Subsequent runs skip the download and start in seconds.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib

import modal

app = modal.App("sheaf-videomae")

# ---------------------------------------------------------------------------
# Persistent volume — cache HuggingFace weights across runs
# ---------------------------------------------------------------------------

_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
_HF_CACHE_DIR = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Image — video extra provides transformers + torch
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["video"])
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)

# ---------------------------------------------------------------------------
# Synthetic frame builder (runs locally in @local_entrypoint)
# ---------------------------------------------------------------------------


def _solid_png(r: int, g: int, b: int, width: int = 224, height: int = 224) -> bytes:
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


def _gradient_clip(
    n_frames: int = 16, base_color: tuple[int, int, int] = (200, 50, 50)
) -> list[str]:
    r0, g0, b0 = base_color
    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        frames.append(
            base64.b64encode(
                _solid_png(
                    min(255, int(r0 + t * 40)),
                    min(255, int(g0 + t * 40)),
                    min(255, int(b0 + t * 40)),
                )
            ).decode()
        )
    return frames


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# VideoAnalyzer Modal class
# ---------------------------------------------------------------------------


@app.cls(
    image=_image,
    gpu="T4",
    min_containers=1,
    volumes={_HF_CACHE_DIR: _hf_cache},
    timeout=600,
)
class VideoAnalyzer:
    @modal.enter()
    def load(self) -> None:
        from sheaf.backends.videomae import VideoMAEBackend

        # Embedding backend
        self.embed_backend = VideoMAEBackend(
            model_name="MCG-NJU/videomae-base",
            task="embedding",
            device="cuda",
            pooling="cls",
        )
        self.embed_backend.load()

        # Classification backend (Kinetics-400 fine-tuned)
        self.clf_backend = VideoMAEBackend(
            model_name="MCG-NJU/videomae-base-finetuned-kinetics",
            task="classification",
            device="cuda",
        )
        self.clf_backend.load()

    @modal.method()
    def embed(self, frames_b64: list[str], normalize: bool = True) -> dict:
        from sheaf.api.video import VideoRequest

        req = VideoRequest(
            model_name="videomae",
            frames_b64=frames_b64,
            task="embedding",
            normalize=normalize,
        )
        return self.embed_backend.predict(req).model_dump(mode="json")

    @modal.method()
    def classify(self, frames_b64: list[str]) -> dict:
        from sheaf.api.video import VideoRequest

        req = VideoRequest(
            model_name="videomae-kinetics",
            frames_b64=frames_b64,
            task="classification",
        )
        return self.clf_backend.predict(req).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    analyzer = VideoAnalyzer()

    # --- Build synthetic clips locally ---
    print("Building synthetic clips...")
    clip_red = _gradient_clip(n_frames=16, base_color=(200, 50, 50))
    clip_blue = _gradient_clip(n_frames=16, base_color=(50, 50, 200))
    clip_red2 = _gradient_clip(n_frames=16, base_color=(210, 60, 40))
    print(f"  {len(clip_red)} frames per clip, 224×224 synthetic PNGs\n")

    # --- Embedding ---
    print("--- VideoMAE-base embeddings (T4, CLS pooling) ---")
    r_red = analyzer.embed.remote(clip_red)
    r_blue = analyzer.embed.remote(clip_blue)
    r_red2 = analyzer.embed.remote(clip_red2)

    emb_red = r_red["embedding"]
    emb_blue = r_blue["embedding"]
    emb_red2 = r_red2["embedding"]

    print(f"Embedding dim : {r_red['dim']}")
    print("\nCosine similarities (L2-normalized):")
    print(f"  red  vs blue  : {cosine(emb_red, emb_blue):.4f}")
    print(f"  red  vs red2  : {cosine(emb_red, emb_red2):.4f}")
    print(f"  blue vs red2  : {cosine(emb_blue, emb_red2):.4f}")
    print("  (red vs red2 should be highest)")

    # --- Classification ---
    print("\n--- VideoMAE Kinetics-400 classification (top-5) ---")
    result = analyzer.classify.remote(clip_red)

    print("\nTop-5 predicted actions:")
    for label, score in zip(result["labels"], result["scores"]):
        bar = "#" * int(score * 40)
        print(f"  {label:30s}  {score:.4f}  {bar}")

    print("\nDone.")
