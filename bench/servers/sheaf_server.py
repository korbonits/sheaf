"""Sheaf benchmark server — Chronos-Bolt-tiny via ModelServer.

Run:
    uv run --extra time-series python bench/servers/sheaf_server.py

Endpoint:
    POST http://127.0.0.1:8000/forecaster/predict

The other two bench servers (raw Ray Serve, BentoML) deploy the same
model with the same batch params; the load generator hits whichever one
is up.
"""

from __future__ import annotations

from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.scheduling.batch import BatchPolicy
from sheaf.spec import ResourceConfig

spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
        "torch_dtype": "float32",
    },
    resources=ResourceConfig(num_cpus=1, replicas=1),
    batch_policy=BatchPolicy(max_batch_size=8, timeout_ms=10),
)

server = ModelServer(models=[spec], host="127.0.0.1", port=8000)
server.run()
print("sheaf server ready on http://127.0.0.1:8000/forecaster/predict")
print("Ctrl-C to stop.")

try:
    import time

    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    server.shutdown()
