# Modal

[Modal](https://modal.com) gives you serverless GPU containers with
no cluster to manage. sheaf-serve ships `ModalServer` as a zero-infra
alternative to the Ray Serve path — same `ModelSpec`, same typed
contracts, the difference is the substrate.

```bash
pip install "sheaf-serve[modal,diffusion]"
```

## Quickstart

```python title="app.py"
import modal
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.modal_server import ModalServer

modal_app = modal.App("flux-serving")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sheaf-serve[diffusion]==0.10.0",
)

server = ModalServer(
    app=modal_app,
    image=image,
    gpu="A10G",
    models=[
        ModelSpec(
            name="flux",
            model_type=ModelType.DIFFUSION,
            backend="flux",
            backend_kwargs={
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "torch_dtype": "bfloat16",
            },
        ),
    ],
)
fastapi_app = server.build()
```

Deploy:

```bash
modal deploy app.py
```

Modal returns a public HTTPS URL. Hit it the same way you would the
Ray Serve path:

```bash
curl -X POST https://<your-app>.modal.run/flux/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_type":"diffusion","model_name":"flux","prompt":"a sheaf"}'
```

## What's the same

- `ModelSpec` — identical contracts; you can swap servers without
  changing call sites.
- The HTTP routes — `/{name}/predict`, `/{name}/health`,
  `/{name}/ready`, `/metrics`.
- LoRA selection, Feast resolution (if you wire it), structured
  logging, OTel tracing, Prometheus metrics.

## What's different

- **No `@serve.batch`.** Modal's container model handles requests one
  at a time per slot by default. `BatchPolicy.bucket_by` is silently
  ignored — there's no batch window to bucket within.
- **Cold-start cost is your problem to manage.** Modal scales to zero
  and back; the first request after idle pays the container start +
  `backend.load()` cost. For latency-sensitive workloads, set
  `keep_warm=N` on the deployment.
- **Concurrency.** `allow_concurrent_inputs > 1` enables in-container
  concurrency, but the same caveats apply as with LoRA on Ray Serve:
  `pipeline.set_adapters` is process-global state. If you crank
  concurrency, shard adapter selection across replicas.

## When to pick this over KubeRay

- You don't want a Kubernetes cluster.
- Your GPU usage is bursty; "scale to zero" is a feature, not a
  liability.
- You're prototyping a generative pipeline and want a public URL in
  ten minutes.

## When to pick KubeRay over this

- You need a steady stream of requests; cold-starts hurt p99.
- You want fine-grained `@serve.batch` tuning.
- Compliance / data residency requires self-hosted infra.

## Reference

- Source: [`src/sheaf/modal_server.py`](https://github.com/korbonits/sheaf/blob/main/src/sheaf/modal_server.py)
- Quickstart example: [`examples/quickstart_modal.py`](https://github.com/korbonits/sheaf/blob/main/examples/quickstart_modal.py)
- LoRA on Modal: [`examples/quickstart_flux_lora_modal.py`](https://github.com/korbonits/sheaf/blob/main/examples/quickstart_flux_lora_modal.py)
