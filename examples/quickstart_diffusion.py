"""FLUX diffusion quickstart: text-to-image generation.

Requirements:
    pip install "sheaf-serve[diffusion]"

    FLUX.1-schnell downloads ~24 GB of weights on first run.
    A CUDA GPU with at least 12 GB VRAM (bfloat16) is recommended.
    On low-VRAM GPUs (<12 GB), pass enable_model_cpu_offload=True.
    CPU inference works but is very slow.

Usage:
    python examples/quickstart_diffusion.py

Demonstrates:
  - FLUX.1-schnell: fast 4-step generation (Apache 2.0)
  - Seed-pinned generation for reproducibility
  - Multiple prompts / batch predict
  - Saving generated images to PNG files
"""

from __future__ import annotations

import base64
import pathlib

from sheaf.api.diffusion import DiffusionRequest
from sheaf.backends.flux import FluxBackend

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load the backend
# ---------------------------------------------------------------------------

print("Loading FLUX.1-schnell (downloads ~24 GB on first run)...")

backend = FluxBackend(
    model_id="black-forest-labs/FLUX.1-schnell",
    device="cuda",  # change to "mps" on Apple Silicon, "cpu" for no GPU
    torch_dtype="bfloat16",  # "float16" on older GPUs; "float32" for CPU
    # enable_model_cpu_offload=True,  # uncomment for <12 GB VRAM
)
backend.load()
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------------

req = DiffusionRequest(
    model_name="flux-schnell",
    prompt="a cat sitting on a moon, digital art, high detail",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=0.0,
    seed=42,
)

print(f"Prompt : {req.prompt!r}")
print(f"Size   : {req.width}×{req.height}  steps={req.num_inference_steps}")
resp = backend.predict(req)

out = OUTPUT_DIR / "flux_single.png"
out.write_bytes(base64.b64decode(resp.image_b64))
print(f"Saved  : {out}  (seed={resp.seed})\n")

# ---------------------------------------------------------------------------
# Seed reproducibility: generate the same image twice
# ---------------------------------------------------------------------------

print("Reproducibility check — two calls with seed=7 should produce identical PNGs...")

req_a = DiffusionRequest(
    model_name="flux-schnell",
    prompt="a snowy forest at dusk",
    seed=7,
)
req_b = DiffusionRequest(
    model_name="flux-schnell",
    prompt="a snowy forest at dusk",
    seed=7,
)

resp_a = backend.predict(req_a)
resp_b = backend.predict(req_b)
match = resp_a.image_b64 == resp_b.image_b64
print(f"Images identical: {match}\n")

# ---------------------------------------------------------------------------
# Batch predict — three prompts in one call
# ---------------------------------------------------------------------------

prompts = [
    "a red apple on a wooden table, photorealistic",
    "an abstract painting of ocean waves, vibrant colors",
    "a futuristic city skyline at night",
]

print(f"Batch of {len(prompts)} prompts...")
reqs = [
    DiffusionRequest(
        model_name="flux-schnell",
        prompt=p,
        height=512,
        width=512,
        num_inference_steps=4,
        seed=i,
    )
    for i, p in enumerate(prompts)
]

results = backend.batch_predict(reqs)
for i, (r, res) in enumerate(zip(reqs, results)):
    out = OUTPUT_DIR / f"flux_batch_{i}.png"
    out.write_bytes(base64.b64decode(res.image_b64))
    print(f"  [{i}] seed={res.seed}  size={res.width}×{res.height}  → {out}")

print("\nDone. Check the outputs/ directory for generated images.")
