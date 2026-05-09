# Docker

The official base image is published to the GitHub Container Registry:

```
ghcr.io/korbonits/sheaf-serve:v0.10.0
ghcr.io/korbonits/sheaf-serve:0.10.0
ghcr.io/korbonits/sheaf-serve:latest
```

It contains:

- Python 3.11 (system Python; no venv — KubeRay's PATH resolution
  expects `/usr/local/bin/ray`)
- `sheaf-serve` core only (no model extras)
- `ray` and `uvicorn`
- Working directory `/workspace`, exposed port `8000`

The image is built and pushed by [`.github/workflows/docker.yml`](https://github.com/korbonits/sheaf/blob/main/.github/workflows/docker.yml)
on every `v*` git tag, gated on real `docker run` smoke + KubeRay
manifest schema validation passing first. We don't push images that
don't actually serve.

## Quickstart — extend the base

The canonical pattern: `FROM` the base image, install the model extras
your deployment actually needs, copy your server script, set `CMD`.

```dockerfile
FROM ghcr.io/korbonits/sheaf-serve:v0.10.0

RUN pip install --no-cache-dir 'sheaf-serve[time-series]==0.10.0'

COPY server.py .

CMD ["python", "server.py"]
```

Where `server.py` is your `ModelServer.run()` script. A complete
example lives at [`examples/docker/`](https://github.com/korbonits/sheaf/tree/main/examples/docker)
in the repo.

```python title="server.py"
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType

server = ModelServer(
    models=[
        ModelSpec(
            name="chronos",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={
                "model_id": "amazon/chronos-bolt-tiny",
                "device_map": "cpu",
                "torch_dtype": "float32",
            },
        ),
    ],
    host="0.0.0.0",  # bind all interfaces inside the container
    port=8000,
)
server.run()

import time
while True:
    time.sleep(3600)  # block; Ctrl-C / SIGTERM stops it
```

## Build + run

```bash
cd examples/docker
docker build -t my-sheaf:dev .
docker run --rm -p 8000:8000 my-sheaf:dev
```

In another shell:

```bash
curl http://localhost:8000/chronos/health
curl -X POST http://localhost:8000/chronos/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_type": "time_series",
    "model_name": "chronos",
    "history": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "horizon": 3,
    "frequency": "1h"
  }'
```

## Why pin the version, not `:latest`

The `:latest` tag tracks main and rolls forward on every release. For
production deploys, **pin the version explicitly** — the same way you
pin `sheaf-serve==0.10.0` in your application code. Bump intentionally,
not silently.

## GPU images

The base image installs CPU-only PyTorch via the model extras. For
GPU workloads, replace the base PyTorch with the CUDA-targeted wheel
in your child image:

```dockerfile
FROM ghcr.io/korbonits/sheaf-serve:v0.10.0

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir 'sheaf-serve[diffusion]==0.10.0'

COPY server.py .
CMD ["python", "server.py"]
```

Run with `--gpus all`:

```bash
docker run --rm --gpus all -p 8000:8000 my-sheaf-gpu:dev
```

## Reference

- Base image source: [Dockerfile](https://github.com/korbonits/sheaf/blob/main/Dockerfile)
- User-facing example: [examples/docker/](https://github.com/korbonits/sheaf/tree/main/examples/docker)
- Publish workflow: [.github/workflows/docker.yml](https://github.com/korbonits/sheaf/blob/main/.github/workflows/docker.yml)
