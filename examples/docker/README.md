# Sheaf in a container

The official base image at `ghcr.io/korbonits/sheaf-serve:vX.Y.Z` installs
`sheaf-serve` core into a slim Python image.  No backend extras, no
entrypoint — it's a foundation you extend.

This directory shows the extension pattern.

## Files

- `Dockerfile` — extends the base image; installs `sheaf-serve[time-series]`
  and copies in `server.py`.
- `server.py` — the deployment entrypoint.  Calls `ModelServer.run()` and
  blocks forever.

## Build

```bash
cd examples/docker
docker build -t my-sheaf:dev .
```

The Dockerfile uses `COPY server.py .`, so the build context must be a
directory that has `server.py` at the root.  Real-world usage: copy this
`Dockerfile` and `server.py` into your own project root and run
`docker build .` there.

## Run

```bash
docker run --rm -p 8000:8000 my-sheaf:dev
```

The first time, the Chronos-Bolt-Tiny weights download from HuggingFace (~30
MB).  Once `[INFO] Application startup complete.` shows in the logs, the
server is ready.

## Test

```bash
curl -s http://localhost:8000/chronos/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/chronos/predict \
    -H 'Content-Type: application/json' \
    -d '{
          "model_type": "time_series",
          "model_name": "chronos",
          "history": [312, 298, 275, 260, 255, 263, 285, 320,
                      368, 402, 421, 435, 442, 438, 430, 425],
          "horizon": 6,
          "frequency": "1h"
        }' | jq '.mean'
```

## GPU images

The official base targets CPU.  For CUDA workloads (FLUX, GraphCast,
SDXL), build a CUDA-based image yourself by swapping the base in the
official `Dockerfile` for an NVIDIA CUDA runtime image, e.g.:

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder
RUN apt-get update && apt-get install --no-install-recommends -y \
        python3.11 python3.11-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# … (rest of the official Dockerfile, adapted)
```

A first-class CUDA image variant is on the v0.10.x roadmap — issue welcome.

## Other deployment shapes

- **Modal** — `pip install 'sheaf-serve[modal]'` and use `ModalServer`.
  No container build needed; Modal handles the image.  See
  `examples/quickstart_diffusion_modal.py` and
  `examples/quickstart_flux_lora_modal.py`.
- **Kubernetes** — see `examples/k8s/` for a KubeRay `RayService` manifest
  that uses this image (or your own derivative).
