"""PointNet 3D point cloud quickstart.

Requirements:
    pip install "sheaf-serve[lidar]"

Usage:
    python examples/quickstart_lidar.py

Demonstrates:
  - PointNet global feature extraction (1024-dim L2-normalized embedding)
  - ModelNet40 classification (40-class softmax over 3D shapes)
  - Cosine similarity between related vs unrelated shape embeddings
  - Checkpoint loading from a saved state dict
  - Synthetic point clouds (unit-sphere normalized, no dataset required)
"""

from __future__ import annotations

import base64
import math

import numpy as np

from sheaf.api.point_cloud import PointCloudRequest
from sheaf.backends.pointnet import PointNetBackend

# ---------------------------------------------------------------------------
# Synthetic point cloud builders
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(seed=42)


def _sphere(n: int = 1024) -> np.ndarray:
    """Uniform random points on the unit sphere surface."""
    pts = _RNG.standard_normal((n, 3)).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _cube(n: int = 1024) -> np.ndarray:
    """Uniform random points on the surface of the unit cube, normalised."""
    pts = _RNG.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    # Snap one coordinate to ±1 per point (face of cube)
    face_axis = _RNG.integers(0, 3, size=n)
    face_sign = _RNG.choice([-1.0, 1.0], size=n)
    for i in range(n):
        pts[i, face_axis[i]] = face_sign[i]
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts / norms).astype(np.float32)


def _cylinder(n: int = 1024, h: float = 2.0) -> np.ndarray:
    """Uniform random points on a cylinder, normalised to unit sphere."""
    theta = _RNG.uniform(0, 2 * math.pi, n).astype(np.float32)
    z = _RNG.uniform(-h / 2, h / 2, n).astype(np.float32)
    x = np.cos(theta)
    y = np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return (pts / norms).astype(np.float32)


def _encode(pts: np.ndarray) -> str:
    return base64.b64encode(pts.tobytes()).decode()


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Build point clouds
# ---------------------------------------------------------------------------

N = 1024
print(f"Building synthetic point clouds ({N} points each)...")
sphere1 = _sphere(N)
sphere2 = _sphere(N)  # different random sample, same shape
cube = _cube(N)
cylinder = _cylinder(N)
print()

# ---------------------------------------------------------------------------
# Load backend (random init — no checkpoint needed for embeddings)
# ---------------------------------------------------------------------------

print("Loading PointNet (random init, pure PyTorch — no download)...")
backend = PointNetBackend(device="cpu")
backend.load()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Embedding task
# ---------------------------------------------------------------------------

print("--- Embedding task (task='embed') ---")

req_s1 = PointCloudRequest(
    model_name="pointnet", points_b64=_encode(sphere1), n_points=N, task="embed"
)
req_s2 = PointCloudRequest(
    model_name="pointnet", points_b64=_encode(sphere2), n_points=N, task="embed"
)
req_cube = PointCloudRequest(
    model_name="pointnet", points_b64=_encode(cube), n_points=N, task="embed"
)
req_cyl = PointCloudRequest(
    model_name="pointnet", points_b64=_encode(cylinder), n_points=N, task="embed"
)

resp_s1 = backend.predict(req_s1)
resp_s2 = backend.predict(req_s2)
resp_cube = backend.predict(req_cube)
resp_cyl = backend.predict(req_cyl)

print(f"Embedding dim : {len(resp_s1.embedding)}")
print("\nCosine similarities (random weights — not meaningful without training):")
print(f"  sphere1 vs sphere2  : {cosine(resp_s1.embedding, resp_s2.embedding):.4f}")
print(f"  sphere1 vs cube     : {cosine(resp_s1.embedding, resp_cube.embedding):.4f}")
print(f"  sphere1 vs cylinder : {cosine(resp_s1.embedding, resp_cyl.embedding):.4f}")
print(f"  cube    vs cylinder : {cosine(resp_cube.embedding, resp_cyl.embedding):.4f}")

# ---------------------------------------------------------------------------
# Classification task (random weights → random predictions)
# ---------------------------------------------------------------------------

print("\n--- Classification task (task='classify', ModelNet40, random weights) ---")

req_cls = PointCloudRequest(
    model_name="pointnet", points_b64=_encode(sphere1), n_points=N, task="classify"
)
resp_cls = backend.predict(req_cls)

print(f"Predicted label : {resp_cls.label!r}")
print(f"Num classes     : {len(resp_cls.scores)}")
top5_idx = sorted(
    range(len(resp_cls.scores)), key=lambda i: resp_cls.scores[i], reverse=True
)[:5]
print("\nTop-5 predicted classes (random weights):")
for rank, idx in enumerate(top5_idx, 1):
    name = resp_cls.label_names[idx]
    score = resp_cls.scores[idx]
    bar = "#" * int(score * 200)
    print(f"  {rank}. {name:15s}  {score:.4f}  {bar}")

# ---------------------------------------------------------------------------
# Custom label set
# ---------------------------------------------------------------------------

print("\n--- Custom 3-class classifier ---")

custom_backend = PointNetBackend(
    num_classes=3,
    label_names=["sphere", "cube", "cylinder"],
    device="cpu",
)
custom_backend.load()

for shape_name, pts in [("sphere", sphere1), ("cube", cube), ("cylinder", cylinder)]:
    req = PointCloudRequest(
        model_name="pointnet-3cls",
        points_b64=_encode(pts),
        n_points=N,
        task="classify",
    )
    resp = custom_backend.predict(req)
    print(f"  {shape_name:8s}  → predicted: {resp.label!r}  (random weights)")

print("\nDone.")
print()
print("To use a trained checkpoint:")
print("  backend = PointNetBackend(checkpoint_path='path/to/model.pth', device='cuda')")
print("  backend.load()")
print("  resp = backend.predict(req)")
