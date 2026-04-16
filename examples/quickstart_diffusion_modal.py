"""FLUX diffusion on Modal — GPU image generation without a local GPU.

Runs FLUX.1-schnell on an A10G via Modal cloud compute.

Prerequisites:
    pip install modal
    modal setup              # authenticate once

    # HuggingFace token (accept FLUX.1-schnell license at hf.co first):
    modal secret create huggingface HF_TOKEN=<your-token>

Run:
    modal run examples/quickstart_diffusion_modal.py

Generated PNGs are written to outputs/ locally.

The model is cached inside the Modal volume after the first run (~24 GB).
Subsequent runs skip the download and start in seconds.
"""

from __future__ import annotations

import base64
import pathlib

import modal

app = modal.App("sheaf-flux")

# ---------------------------------------------------------------------------
# Persistent volume — cache HuggingFace weights across runs
# ---------------------------------------------------------------------------

_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
_HF_CACHE_DIR = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Image — diffusion extra provides diffusers + torch + transformers
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["diffusion"])
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)

# ---------------------------------------------------------------------------
# FluxGenerator Modal class
# ---------------------------------------------------------------------------


@app.cls(
    image=_image,
    gpu="T4",
    min_containers=1,
    volumes={_HF_CACHE_DIR: _hf_cache},
    timeout=600,
)
class FluxGenerator:
    @modal.enter()
    def load(self) -> None:
        from sheaf.backends.flux import FluxBackend

        # tiny-flux-pipe is an ungated ~100 MB model used to verify the full
        # code path without a 24 GB download.  Swap for
        # "black-forest-labs/FLUX.1-schnell" (needs HF_TOKEN + A10G) for real images.
        self.backend = FluxBackend(
            model_id="hf-internal-testing/tiny-flux-pipe",
            device="cuda",
            torch_dtype="bfloat16",
        )
        self.backend.load()

    @modal.method()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
    ) -> dict:
        from sheaf.api.diffusion import DiffusionRequest

        req = DiffusionRequest(
            model_name="flux-schnell",
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return self.backend.predict(req).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    out_dir = pathlib.Path("outputs")
    out_dir.mkdir(exist_ok=True)

    generator = FluxGenerator()

    # --- Single generation ---
    print("\n--- FLUX.1-schnell (A10G, bfloat16) ---")

    prompt = "a cat sitting on a moon, digital art, cinematic lighting"
    print(f"Prompt : {prompt!r}")
    print("Size   : 64×64  steps=2  guidance=0.0  seed=42")

    result = generator.generate.remote(
        prompt=prompt,
        height=64,
        width=64,
        num_inference_steps=2,
        guidance_scale=0.0,
        seed=42,
    )

    p = out_dir / "flux_modal_single.png"
    p.write_bytes(base64.b64decode(result["image_b64"]))
    print(f"Saved  : {p}  (seed={result['seed']})")

    # --- Reproducibility check ---
    print("\n--- Reproducibility check (seed=7 x2) ---")

    r1 = generator.generate.remote(
        prompt="a snowy forest at dusk, oil painting",
        seed=7,
    )
    r2 = generator.generate.remote(
        prompt="a snowy forest at dusk, oil painting",
        seed=7,
    )
    print(f"Images identical: {r1['image_b64'] == r2['image_b64']}")

    # --- Batch: three prompts ---
    print("\n--- Batch (3 prompts, 512×512) ---")

    prompts = [
        "a red apple on a wooden table, photorealistic",
        "an abstract painting of ocean waves, vibrant colors",
        "a futuristic city skyline at night, neon lights",
    ]

    calls = [
        generator.generate.remote(
            prompt=p,
            height=64,
            width=64,
            num_inference_steps=2,
            seed=i,
        )
        for i, p in enumerate(prompts)
    ]

    for i, (p, res) in enumerate(zip(prompts, calls)):
        out = out_dir / f"flux_modal_batch_{i}.png"
        out.write_bytes(base64.b64decode(res["image_b64"]))
        print(f"  [{i}] seed={res['seed']}  → {out}")
        print(f"       {p!r}")

    print("\nDone. Check outputs/ for generated images.")
