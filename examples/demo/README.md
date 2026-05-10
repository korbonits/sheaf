# Live demo on Modal

This directory holds the source for the public sheaf-serve demo deployment
at https://korbonits--sheaf-demo-modalserver---init----locals---serve.modal.run.

It's a one-file `ModalServer` running `amazon/chronos-bolt-tiny` on a
CPU-only Modal container with one warm replica. The model is small enough
(~80 MB) that cold-start fits comfortably inside Modal's free tier.

## Deploy from this repo

```bash
# from repo root
pip install modal
modal setup    # authenticate once

# optional but recommended — a Modal secret so the container can fetch
# from Hugging Face without rate limiting
modal secret create huggingface HF_TOKEN=<your-hf-token>

modal deploy examples/demo/app.py
```

Modal prints the public URL once the image is built.

## Why this file looks short

The interesting bit is *that the file is short*. The whole demo is one
`ModelSpec` plus a `ModalServer` instance — exactly the same shape you'd
use to serve the model locally with `ModelServer`. Same typed contracts,
same routes, same observability.

Switching this demo to GPU (`gpu="A10G"`), adding a second model, or
flipping to scale-to-zero (`min_containers=0`) are all one-line edits.

## Notes on the image

The image installs `sheaf-serve` (core), `chronos-forecasting`, and
`cloudpickle` directly. Two notes:

1. We don't use the `[time-series]` extra because it pulls in TimesFM,
   which has a transitive `paxml → lingvo` dep that lacks Python 3.12
   wheels. The demo only needs chronos, so we install chronos directly.
2. `cloudpickle` is explicit because `ModalServer._serve()` deserialises
   the spec bundle with it. It's usually pulled in transitively via Ray,
   but pinning it here makes the dependency visible.
