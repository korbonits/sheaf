"""BentoML baseline — same model, same batch params, BentoML's own runtime.

Run:
    uv pip install bentoml
    cd bench/servers
    bentoml serve bentoml_server:svc --host 127.0.0.1 --port 8000

Endpoint (BentoML mounts service routes at /; the load generator's
--url flag points at the right path per server):
    POST http://127.0.0.1:8000/predict

Notes:
- BentoML's batching API is `@bentoml.api(batchable=True, ...)`.  We set
  `max_batch_size=8` and `max_latency_ms=10` to match the other two
  servers' BatchPolicy/serve.batch params.
- Service is defined at module level so `bentoml serve <module>:svc`
  picks it up.  Loading happens once when BentoML imports the module.
"""

from __future__ import annotations

import os
from typing import Any

import bentoml  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]
from chronos import BaseChronosPipeline  # ty: ignore[unresolved-import]
from pydantic import BaseModel

# Worker count parameterised — apples-to-apples with sheaf/Ray Serve replicas.
# Each worker is a separate process; BentoML reverse-proxies via uvicorn.
_WORKERS = int(os.environ.get("BENCH_REPLICAS", "1"))


class Request(BaseModel):
    history: list[float]
    horizon: int = 12


class Response(BaseModel):
    horizon: int
    mean: list[float]


@bentoml.service(
    name="forecaster",
    workers=_WORKERS,
    resources={"cpu": "1"},
    traffic={"timeout": 60},
)
class Forecaster:
    def __init__(self) -> None:
        self._pipeline: Any = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=8,
        max_latency_ms=10,
        route="/predict",
    )
    async def predict(self, requests: list[Request]) -> list[Response]:
        contexts = [torch.tensor(r.history, dtype=torch.float32) for r in requests]
        horizon = requests[0].horizon
        forecast = self._pipeline.predict(inputs=contexts, prediction_length=horizon)
        forecast_np = forecast.numpy()
        return [
            Response(horizon=r.horizon, mean=forecast_np[i][4].tolist())
            for i, r in enumerate(requests)
        ]


svc = Forecaster
