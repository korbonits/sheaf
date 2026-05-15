"""Raw Ray Serve baseline — same model, same batch params, no sheaf.

Hand-rolled @serve.deployment + @serve.ingress wrapping ChronosBoltPipeline
directly.  No sheaf import; this is the apples-to-apples comparison for
the typed-contract overhead claim.

Run:
    uv run --extra time-series python bench/servers/ray_serve_server.py

Endpoint (same path as the sheaf server, for load-generator parity):
    POST http://127.0.0.1:8000/forecaster/predict
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch  # ty: ignore[unresolved-import]
from chronos import BaseChronosPipeline  # ty: ignore[unresolved-import]
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

# Replica count parameterised — see sheaf_server.py for rationale.
_REPLICAS = int(os.environ.get("BENCH_REPLICAS", "1"))


# Same input/output shape as sheaf's TimeSeriesRequest/Response.  We
# hand-roll a minimal version here so the wire format is identical and
# the load generator's payload works against any of the three servers.
class Request(BaseModel):
    history: list[float]
    horizon: int = 12


class Response(BaseModel):
    horizon: int
    mean: list[float]


app = FastAPI()


@serve.deployment(num_replicas=_REPLICAS, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class Forecaster:
    def __init__(self) -> None:
        self._pipeline: Any = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    @app.get("/health")
    def health(self) -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=Response)
    async def predict(self, request: Request) -> Response:
        return await self._batch(request)

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.01)
    async def _batch(self, requests: list[Request]) -> list[Response]:
        contexts = [torch.tensor(r.history, dtype=torch.float32) for r in requests]
        horizon = requests[0].horizon
        forecast = self._pipeline.predict(inputs=contexts, prediction_length=horizon)
        forecast_np = forecast.numpy()
        # Bolt returns [batch, 9, horizon] at quantiles [0.1..0.9]; index 4 = median
        return [
            Response(horizon=r.horizon, mean=forecast_np[i][4].tolist())
            for i, r in enumerate(requests)
        ]


# Match sheaf's URL: POST /forecaster/predict + GET /forecaster/health
serve.start(http_options={"host": "127.0.0.1", "port": 8000})
serve.run(Forecaster.bind(), name="forecaster", route_prefix="/forecaster")
print("raw Ray Serve baseline ready on http://127.0.0.1:8000/forecaster/predict")
print("Ctrl-C to stop.")

try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    serve.shutdown()
