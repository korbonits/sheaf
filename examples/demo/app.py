"""Public sheaf-serve live demo on Modal.

A single-deployment ModalServer exposing chronos-bolt-tiny over HTTP.
CPU-only so it runs cheaply (~free on Modal's free tier with min_containers=1).
The route shape mirrors the docs:

    GET  /chronos/health
    GET  /chronos/ready
    POST /chronos/predict     (TimeSeriesRequest body)
    POST /chronos/stream      (SSE)

Deploy:
    modal deploy examples/demo/app.py

Modal returns a stable URL like:
    https://<workspace>--sheaf-demo-serve.modal.run

Hit it:
    curl https://<workspace>--sheaf-demo-serve.modal.run/chronos/health
    curl -X POST https://<workspace>--sheaf-demo-serve.modal.run/chronos/predict \\
        -H 'Content-Type: application/json' \\
        -d '{
              "model_type": "time_series",
              "model_name": "chronos",
              "history": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
              "horizon": 3,
              "frequency": "1h",
              "output_mode": "mean"
            }'
"""

from __future__ import annotations

import modal

from sheaf import ModalServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.spec import ResourceConfig

# Image with the time-series extra so chronos-forecasting is available
# inside the container.  Pin to the released version on PyPI so demo
# deploys mirror what users would install.
# The [time-series] extra pulls in TimesFM, whose paxml→lingvo
# transitive dep has no 3.12 wheels.  This demo only uses chronos, so
# install chronos-forecasting directly and skip TimesFM.
#
# cloudpickle is explicit because ModalServer's _serve() function
# deserialises the spec bundle with it; sheaf-serve depends on ray which
# usually pulls in cloudpickle transitively, but pinning it here makes
# the dep visible and immune to future ray packaging changes.
_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("sheaf-serve==0.10.0")
    .pip_install("chronos-forecasting>=1.0.0")
    .pip_install("cloudpickle>=3.0.0")
)

_spec = ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
        "torch_dtype": "float32",
    },
    resources=ResourceConfig(num_cpus=1, replicas=1),
)

# min_containers=1 keeps one warm container so demo links don't pay a
# cold-start tax.  Scale-to-zero is min_containers=0; switch to that if
# the demo gets enough traffic that one container isn't enough and Modal
# bills it during idle.
_server = ModalServer(
    models=[_spec],
    app_name="sheaf-demo",
    image=_image,
    gpu=None,  # CPU — chronos-bolt-tiny is small enough
    min_containers=1,
)

#: Module-level binding the Modal CLI looks for.
app = _server.app
