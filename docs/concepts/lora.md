# LoRA adapter multiplexing

Diffusion backends (FLUX, SDXL) ship a working LoRA path. Register a
named adapter registry on the deployment; clients select per request
by name; sheaf-serve handles the per-request adapter activation
inside the batch window so multiple selections share a single
deployment without racing the diffusers pipeline's process-global
state.

```bash
pip install "sheaf-serve[diffusion]"
```

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.lora import LoRAAdapter, LoRAConfig

spec = ModelSpec(
    name="flux",
    model_type=ModelType.DIFFUSION,
    backend="flux",
    backend_kwargs={
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "torch_dtype": "bfloat16",
    },
    lora=LoRAConfig(
        adapters={
            "sketch": LoRAAdapter(source="hf:alvarobartt/flux-sketch-lora", weight=0.8),
            "realism": LoRAAdapter(source="hf:org/flux-realism-lora", weight=1.0),
            "watercolour": LoRAAdapter(source="/path/to/local/wc.safetensors"),
        },
        default="sketch",
    ),
)
```

## Per-request selection

```json
{
  "model_type": "diffusion",
  "model_name": "flux",
  "prompt": "a sheaf, mathematical concept",
  "adapters": ["realism"],
  "adapter_weights": [0.6]
}
```

- `adapters` — list of names from `LoRAConfig.adapters`. Empty list
  disables LoRA for that request (returns to base model).
- `adapter_weights` — optional override. Falls back to per-adapter
  `LoRAAdapter.weight` from the spec, then to `1.0`.
- Specifying multiple adapters composes them at the listed weights.

A request with an unknown adapter name returns `422 Unprocessable
Entity`. A request that specifies `adapters` against a deployment
without `lora` configured also returns 422.

## Adapter source format

`LoRAAdapter.source` accepts:

| Form | Meaning |
|---|---|
| `hf:org/repo` | HuggingFace Hub repo, default weight file |
| `hf:org/repo:weights.safetensors` | HF repo, pinned weight file |
| `/abs/path/to/lora.safetensors` | local file path |

The colon-separated form mirrors `diffusers.load_lora_weights`'s
internal contract — anyone who knows diffusers shouldn't have to
learn a new schema.

## Bucket-by-resolved-adapter (automatic)

`pipeline.set_adapters` on diffusers is process-global state — two
concurrent requests inside the same batch window with different
adapters would race. When `ModelSpec.lora` is set, the deployment
**automatically** groups requests by their resolved
`(adapter_names, weights)` tuple and dispatches each group as its own
`async_batch_predict` call, with `set_active_adapters` called once
per group.

Two requests that resolve to the same selection share a bucket. For
example: a request with empty `adapters` + spec default `"sketch"`
shares a bucket with `adapters=["sketch"]` and the same default
weight.

This bucketing replaces `BatchPolicy.bucket_by` for v1; the spec
validator rejects setting both.

## Loaded once at deploy time

`backend.load_adapters(adapters)` is called exactly once by
`_SheafDeployment.__init__` after `backend.load()`. Hot-add at
runtime is intentionally out of scope — adding a new adapter requires
`ModelServer.update(spec)` (which performs Ray Serve's standard
rolling deployment update). The cleaner alternative — drain in-flight
requests before adding — defeats the point of "no downtime"; the
risk-vs-value of hot-add is poor for v1.

## Modal & streaming caveats

- **Modal**: `ModalServer._build_asgi_app` calls `set_active_adapters`
  inline before `backend.async_predict`. Modal's default concurrency is
  1 per container slot, so this is safe. Cranking `allow_concurrent_inputs > 1`
  without sharding selection across replicas creates the same race as
  Ray Serve does without bucket-by-adapter.
- **Streaming**: `POST /{name}/stream` calls `set_active_adapters`
  inline per request. Streaming is per-request by design and benefits
  little from concurrency; the operational rule is one stream at a
  time per replica when LoRA is in use.

## Reference

Full schema in the [LoRA API reference](../api-reference/lora.md).
