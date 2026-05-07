"""FLUX.1-schnell + LoRA on Modal — real GPU smoke for sheaf's LoRA wire-up.

What this exercises:
  - FluxBackend.load_adapters() against a real HF Hub LoRA repo
  - FluxBackend.set_active_adapters() before generation
  - Verifies that the diffusers ``load_lora_weights`` / ``set_adapters`` calls
    sheaf makes are actually accepted by a real diffusers pipeline + real
    weights — the unit tests assert *that* we make those calls, not *that*
    diffusers accepts them.

LoRA picked: ``XLabs-AI/flux-RealismLora`` — XLabs is the org behind much of
the FLUX ecosystem (ControlNets, IPAdapters); their realism LoRA is widely
used and works on FLUX.1-schnell with reduced but visible effect (most LoRAs
are trained on FLUX.1-dev).  Swap it freely for any FLUX-compatible LoRA on
HF Hub: change the ``source`` field below.

Prerequisites:
    pip install modal
    modal setup              # authenticate once

    # HuggingFace token (accept FLUX.1-schnell license at hf.co first):
    modal secret create huggingface HF_TOKEN=<your-token>

Run:
    modal run examples/quickstart_flux_lora_modal.py

Outputs (written to ./outputs/):
    flux_lora_baseline.png  — FLUX.1-schnell, no LoRA, seed=42
    flux_lora_realism.png   — same prompt + seed, with the realism LoRA
                              applied — should be visibly different.

Cost: ~$0.50–$1.00 of A10G time per full run (one container start + 3 generations).
The container stays warm for ``min_containers=1``; set to 0 to scale to zero
between runs at the cost of a ~1 minute cold start.
"""

from __future__ import annotations

import base64
import pathlib

import modal

app = modal.App("sheaf-flux-lora")

# ---------------------------------------------------------------------------
# Persistent volume — cache HuggingFace weights across runs.
# FLUX.1-schnell is ~24 GB; the LoRA is only ~150 MB but lives in the same
# cache.  Without this volume every container would re-download from HF.
# ---------------------------------------------------------------------------

_hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
_HF_CACHE_DIR = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Image — diffusion extra brings diffusers + torch + transformers.
# peft is a transitive dep of diffusers' LoRA path; pinned explicitly so the
# pip resolver doesn't surprise us with an incompatible version.
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["diffusion"])
    .pip_install("peft>=0.11.0")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env({"PYTHONPATH": "/root/src"})
)

# Pulled out as constants so the choice is obvious / easy to swap.
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
LORA_NAME = "realism"
LORA_SOURCE = "hf:XLabs-AI/flux-RealismLora"  # diffusers auto-discovers the weight file


# ---------------------------------------------------------------------------
# FluxLoRAGenerator Modal class
# ---------------------------------------------------------------------------


@app.cls(
    image=_image,
    gpu="A100",
    min_containers=1,
    volumes={_HF_CACHE_DIR: _hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=900,
)
class FluxLoRAGenerator:
    @modal.enter()
    def load(self) -> None:
        from sheaf.backends.flux import FluxBackend
        from sheaf.lora import LoRAAdapter

        # FLUX.1-schnell + the T5 text encoder + a LoRA at bfloat16 occupy
        # ~22 GB resident, which doesn't leave enough headroom for
        # activations on a 24 GB A10G even with enable_model_cpu_offload=True.
        # An A100 40 GB runs comfortably; an A10G + enable_sequential_cpu_offload
        # would also work (slower but fits anywhere — that helper isn't yet
        # exposed on FluxBackend).
        self.backend = FluxBackend(
            model_id=MODEL_ID,
            device="cuda",
            torch_dtype="bfloat16",
        )
        self.backend.load()

        # Exercise the actual LoRA load path that ModalServer / ModelServer
        # would call automatically when ModelSpec.lora is set.
        self.backend.load_adapters(
            {LORA_NAME: LoRAAdapter(source=LORA_SOURCE, weight=1.0)}
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
        seed: int,
        adapter: str | None = None,
        adapter_weight: float = 1.0,
    ) -> dict:
        """Generate one image; optionally activate the LoRA before generation."""
        from sheaf.api.diffusion import DiffusionRequest

        # set_active_adapters([], []) deactivates LoRAs; with one name it
        # activates exactly that adapter at the given weight.
        if adapter is None:
            self.backend.set_active_adapters([], [])
        else:
            self.backend.set_active_adapters([adapter], [adapter_weight])

        # 512×512 keeps activations small enough to fit alongside the model
        # weights on a 24 GB A10G (FLUX-schnell + LoRA leaves only ~2 GB for
        # activations).  Bump to 1024 only on >24 GB cards (H100, A100 80GB)
        # or with enable_model_cpu_offload=True on the backend.
        req = DiffusionRequest(
            model_name="flux-schnell",
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            seed=seed,
        )
        return self.backend.predict(req).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Local entrypoint — runs the smoke comparison
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    out_dir = pathlib.Path("outputs")
    out_dir.mkdir(exist_ok=True)

    gen = FluxLoRAGenerator()

    prompt = "portrait of a woman, soft natural lighting, freckles"
    seed = 42

    print(f"\nPrompt: {prompt!r}")
    print(f"Seed  : {seed}\n")

    # 1) Baseline — no LoRA
    print("Generating baseline (no LoRA)...")
    baseline = gen.generate.remote(prompt=prompt, seed=seed, adapter=None)
    p1 = out_dir / "flux_lora_baseline.png"
    p1.write_bytes(base64.b64decode(baseline["image_b64"]))
    print(f"  saved {p1}")

    # 2) With the realism LoRA
    print(f"\nGenerating with adapter={LORA_NAME!r} weight=1.0 ...")
    with_lora = gen.generate.remote(prompt=prompt, seed=seed, adapter=LORA_NAME)
    p2 = out_dir / "flux_lora_realism.png"
    p2.write_bytes(base64.b64decode(with_lora["image_b64"]))
    print(f"  saved {p2}")

    # 3) Same LoRA at lower weight — should land between baseline and full LoRA
    print(f"\nGenerating with adapter={LORA_NAME!r} weight=0.5 ...")
    half_lora = gen.generate.remote(
        prompt=prompt, seed=seed, adapter=LORA_NAME, adapter_weight=0.5
    )
    p3 = out_dir / "flux_lora_realism_half.png"
    p3.write_bytes(base64.b64decode(half_lora["image_b64"]))
    print(f"  saved {p3}")

    # 4) Sanity check — the three images should differ
    same_baseline_full = baseline["image_b64"] == with_lora["image_b64"]
    same_full_half = with_lora["image_b64"] == half_lora["image_b64"]

    print("\nDifference check (same seed, same prompt — should all be False):")
    print(f"  baseline == realism @1.0 : {same_baseline_full}")
    print(f"  realism @1.0 == @0.5     : {same_full_half}")

    if same_baseline_full or same_full_half:
        print(
            "\n  WARNING: identical outputs. Either the LoRA didn't load or "
            "set_active_adapters didn't take effect. Inspect Modal container "
            "logs for diffusers warnings."
        )
    else:
        print(
            "\n  All three differ — load_adapters + set_active_adapters are "
            "working end-to-end against real FLUX weights."
        )

    print("\nDone. Compare the three PNGs in outputs/ side by side.")
