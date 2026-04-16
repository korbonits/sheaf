"""Vision embedding quickstart: DINOv2 and OpenCLIP.

Requirements:
    pip install "sheaf-serve[vision]"

Usage:
    python examples/quickstart_vision.py

Demonstrates:
  - DINOv2 image embeddings with CLS and mean pooling
  - OpenCLIP image embeddings and text embeddings
  - Cross-modal similarity: ranking images by a text query
"""

from __future__ import annotations

import base64
import math
import urllib.request

from sheaf.api.embedding import EmbeddingRequest
from sheaf.backends.dinov2 import DINOv2Backend
from sheaf.backends.open_clip import OpenCLIPBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Three small public-domain images from Wikimedia Commons (JPEG, ~10–30 KB each)
IMAGE_URLS = {
    "cat": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg"
    ),
    "dog": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"
    ),
    "bird": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    ),
}


def fetch_image_b64(url: str) -> str:
    """Download an image and return it as a base64 string."""
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
        return base64.b64encode(resp.read()).decode()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Download images
# ---------------------------------------------------------------------------

print("Downloading images...")
images: dict[str, str] = {}
for label, url in IMAGE_URLS.items():
    images[label] = fetch_image_b64(url)
    print(f"  {label:6s}  {len(images[label]) * 3 // 4 // 1024} KB")

labels = list(images.keys())
images_b64 = [images[k] for k in labels]

# ---------------------------------------------------------------------------
# DINOv2 — image embeddings (CLS pooling, then mean pooling)
# ---------------------------------------------------------------------------

print("\n--- DINOv2 (facebook/dinov2-base, 768-dim) ---")
print("Loading model (downloads ~330 MB on first run)...")

dino = DINOv2Backend(model_name="facebook/dinov2-base", device="cpu", pooling="cls")
dino.load()
print("Model loaded.")

req = EmbeddingRequest(model_name="dinov2", images_b64=images_b64, normalize=True)
resp = dino.predict(req)

print(f"\nEmbedding dim : {resp.dim}")
print("\nCosine similarities (CLS pooling, L2-normalized):")
embs = resp.embeddings
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        sim = cosine(embs[i], embs[j])
        print(f"  {labels[i]:6s} vs {labels[j]:6s}  {sim:.4f}")

# Mean pooling variant
dino_mean = DINOv2Backend(
    model_name="facebook/dinov2-base", device="cpu", pooling="mean"
)
dino_mean.load()
req_mean = EmbeddingRequest(model_name="dinov2", images_b64=images_b64, normalize=True)
resp_mean = dino_mean.predict(req_mean)

print("\nCosine similarities (mean pooling, L2-normalized):")
embs_mean = resp_mean.embeddings
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        sim = cosine(embs_mean[i], embs_mean[j])
        print(f"  {labels[i]:6s} vs {labels[j]:6s}  {sim:.4f}")

# ---------------------------------------------------------------------------
# OpenCLIP — image embeddings + text embeddings + cross-modal retrieval
# ---------------------------------------------------------------------------

print("\n--- OpenCLIP (ViT-B-32 / openai, 512-dim) ---")
print("Loading model (downloads ~350 MB on first run)...")

clip = OpenCLIPBackend(model_name="ViT-B-32", pretrained="openai", device="cpu")
clip.load()
print("Model loaded.")

# Image embeddings
img_req = EmbeddingRequest(
    model_name="open-clip", images_b64=images_b64, normalize=True
)
img_resp = clip.predict(img_req)

print(f"\nEmbedding dim : {img_resp.dim}")
print("\nImage–image cosine similarities:")
img_embs = img_resp.embeddings
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        sim = cosine(img_embs[i], img_embs[j])
        print(f"  {labels[i]:6s} vs {labels[j]:6s}  {sim:.4f}")

# Text embeddings + cross-modal retrieval
queries = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
txt_req = EmbeddingRequest(model_name="open-clip", texts=queries, normalize=True)
txt_resp = clip.predict(txt_req)

print("\nCross-modal retrieval (text query → best matching image):")
for q, txt_emb in zip(queries, txt_resp.embeddings):
    sims = [(labels[i], cosine(txt_emb, img_embs[i])) for i in range(len(labels))]
    sims.sort(key=lambda x: x[1], reverse=True)
    ranking = "  ".join(f"{lbl}={s:.3f}" for lbl, s in sims)
    print(f"  {q!r:30s}  →  {ranking}")
