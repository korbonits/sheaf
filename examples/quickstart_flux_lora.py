"""FLUX + LoRA adapter multiplexing quickstart.

Requirements:
    pip install "sheaf-serve[diffusion]"

    Plus the LoRA weights themselves — either local .safetensors files or
    HuggingFace Hub repos.  The references below assume two adapters; swap
    them for your own paths or HF repos.

Usage:
    python examples/quickstart_flux_lora.py

Demonstrates:
  - Declaring multiple LoRA adapters on a single deployment via
    ``ModelSpec.lora = LoRAConfig(...)``
  - Local-path and HuggingFace Hub adapter sources
  - Per-request adapter selection via ``DiffusionRequest.adapters``
  - Per-request weight overrides via ``DiffusionRequest.adapter_weights``
  - Adapter fusion: applying two LoRAs together with custom weights
  - Bucket-by-adapter sub-batching: the deployment dispatches each unique
    adapter selection in a single ``set_active_adapters`` call per Ray
    Serve batch window, so concurrent requests with different adapters
    don't race on the diffusers pipeline state.

Note: this example uses ``ModelServer`` (Ray Serve) so the bucket-by-adapter
batching is exercised.  For Modal, use ``ModalServer`` with the same spec.
"""

from __future__ import annotations

from sheaf import ModelServer
from sheaf.api.base import ModelType
from sheaf.lora import LoRAAdapter, LoRAConfig
from sheaf.scheduling.batch import BatchPolicy
from sheaf.spec import ModelSpec, ResourceConfig

# ---------------------------------------------------------------------------
# Declare adapters
# ---------------------------------------------------------------------------
# Two adapters: one local, one on HuggingFace Hub.  Replace the paths below
# with your own LoRA weights — these names are illustrative.

LORA_CONFIG = LoRAConfig(
    adapters={
        # Local path — accepts a .safetensors file or a directory.
        "sketch": LoRAAdapter(
            source="/path/to/sketch_style.safetensors",
            weight=0.8,  # default weight when this adapter is selected
        ),
        # HuggingFace Hub reference — "hf:org/repo" or "hf:org/repo:weight_file"
        # to pin a specific weight file inside a multi-adapter repo.
        "watercolor": LoRAAdapter(
            source="hf:user/flux-watercolor-lora",
            weight=1.0,
        ),
    },
    default="sketch",  # applied when a request omits ``adapters``
)

# ---------------------------------------------------------------------------
# Build the spec
# ---------------------------------------------------------------------------
# When ``lora`` is set on the spec, requests are automatically grouped by
# their resolved (names, weights) selection inside each Ray Serve batch
# window.  ``set_active_adapters`` is called once per group — you do NOT
# need (and cannot use) ``BatchPolicy.bucket_by`` alongside it.

spec = ModelSpec(
    name="flux-loras",
    model_type=ModelType.DIFFUSION,
    backend="flux",
    backend_kwargs={
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "device": "cuda",
        "torch_dtype": "bfloat16",
    },
    resources=ResourceConfig(num_gpus=1),
    batch_policy=BatchPolicy(max_batch_size=8, timeout_ms=50),
    lora=LORA_CONFIG,
)

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = ModelServer(models=[spec])
    server.run()
    print("FLUX-LoRAs serving at http://localhost:8000/flux-loras/predict")
    print()
    print("Example requests:")
    print()
    print("# Default adapter (sketch, weight=0.8)")
    print("""curl -s http://localhost:8000/flux-loras/predict \\
  -H 'Content-Type: application/json' \\
  -d '{
        "model_type": "diffusion",
        "model_name": "flux-schnell",
        "prompt": "a cat on the moon"
      }' | jq .""")
    print()
    print("# Override default with watercolor")
    print("""curl -s http://localhost:8000/flux-loras/predict \\
  -H 'Content-Type: application/json' \\
  -d '{
        "model_type": "diffusion",
        "model_name": "flux-schnell",
        "prompt": "a cat on the moon",
        "adapters": ["watercolor"]
      }' | jq .""")
    print()
    print("# Fusion: apply both adapters with custom weights")
    print("""curl -s http://localhost:8000/flux-loras/predict \\
  -H 'Content-Type: application/json' \\
  -d '{
        "model_type": "diffusion",
        "model_name": "flux-schnell",
        "prompt": "a cat on the moon",
        "adapters": ["sketch", "watercolor"],
        "adapter_weights": [0.5, 0.7]
      }' | jq .""")
