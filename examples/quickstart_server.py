"""Sheaf multi-model server quickstart.

Deploys three heterogeneous model backends on the same Ray Serve instance
and queries each via HTTP — the core value proposition of sheaf.

Models used (all CPU-friendly, no GPU required):
  - time_series  : amazon/chronos-bolt-tiny  (~80 MB)
  - tabular      : TabPFN v2                 (requires TABPFN_TOKEN env var)
  - audio        : openai/whisper-tiny       (~150 MB)

Usage:
    pip install "sheaf-serve[time-series,tabular,audio]"
    export TABPFN_TOKEN=<your-token>   # https://ux.priorlabs.ai
    python examples/quickstart_server.py

Each deployment gets its own URL:
    POST http://127.0.0.1:8000/forecaster/predict
    POST http://127.0.0.1:8000/classifier/predict
    POST http://127.0.0.1:8000/transcriber/predict
"""

from __future__ import annotations

import base64
import pathlib
import time

import requests

from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.spec import ResourceConfig

# ---------------------------------------------------------------------------
# 1. Declare what to serve
# ---------------------------------------------------------------------------

forecaster_spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
        "torch_dtype": "float32",
    },
    resources=ResourceConfig(num_cpus=1, replicas=1),
)

classifier_spec = ModelSpec(
    name="classifier",
    model_type=ModelType.TABULAR,
    backend="tabpfn",
    resources=ResourceConfig(num_cpus=1, replicas=1),
)

transcriber_spec = ModelSpec(
    name="transcriber",
    model_type=ModelType.AUDIO,
    backend="whisper",
    backend_kwargs={"model_size": "tiny"},
    resources=ResourceConfig(num_cpus=1, replicas=1),
)

# ---------------------------------------------------------------------------
# 2. Start the server
# ---------------------------------------------------------------------------

server = ModelServer(
    models=[forecaster_spec, classifier_spec, transcriber_spec],
    host="127.0.0.1",
    port=8000,
)

print("Starting Ray Serve... (first run downloads model weights)")
server.run()

BASE = "http://127.0.0.1:8000"


def _wait_ready(name: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/{name}/health", timeout=2)
            if r.status_code == 200:
                print(f"  {name}: ready")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError(f"{name} did not become ready within {timeout}s")


print("\nWaiting for deployments...")
for deployment in ("forecaster", "classifier", "transcriber"):
    _wait_ready(deployment)

# ---------------------------------------------------------------------------
# 3. Time series forecast
# ---------------------------------------------------------------------------

print("\n--- Time Series Forecast ---")

ts_payload = {
    "model_type": "time_series",
    "model_name": "forecaster",
    "history": [312, 298, 275, 260, 255, 263, 285, 320, 368, 402, 421, 435],
    "horizon": 6,
    "frequency": "1h",
    "output_mode": "quantiles",
    "quantile_levels": [0.1, 0.5, 0.9],
}

r = requests.post(f"{BASE}/forecaster/predict", json=ts_payload)
r.raise_for_status()
ts = r.json()

print(f"{'Hour':>5}  {'P10':>7}  {'Median':>7}  {'P90':>7}")
print("-" * 35)
p10 = ts["quantiles"]["0.1"]
p50 = ts["quantiles"]["0.5"]
p90 = ts["quantiles"]["0.9"]
for i in range(ts["horizon"]):
    print(f"{i + 1:>5}  {p10[i]:>7.1f}  {p50[i]:>7.1f}  {p90[i]:>7.1f}")

# ---------------------------------------------------------------------------
# 4. Tabular classification
# ---------------------------------------------------------------------------

print("\n--- Tabular Classification (Iris) ---")

# Iris dataset: 3 features (sepal_length, sepal_width, petal_length)
# Context = 6 labeled examples (2 per class); query = 2 unseen rows
tab_payload = {
    "model_type": "tabular",
    "model_name": "classifier",
    "context_X": [
        [5.1, 3.5, 1.4],  # setosa
        [4.9, 3.0, 1.4],  # setosa
        [7.0, 3.2, 4.7],  # versicolor
        [6.4, 3.2, 4.5],  # versicolor
        [6.3, 3.3, 6.0],  # virginica
        [5.8, 2.7, 5.1],  # virginica
    ],
    "context_y": [0, 0, 1, 1, 2, 2],
    "query_X": [
        [5.0, 3.4, 1.5],  # likely setosa
        [6.1, 2.8, 4.7],  # likely versicolor
    ],
    "task": "classification",
    "output_mode": "probabilities",
}

r = requests.post(f"{BASE}/classifier/predict", json=tab_payload)
r.raise_for_status()
tab = r.json()

species = {0: "setosa", 1: "versicolor", 2: "virginica"}
for i, (pred, probs) in enumerate(zip(tab["predictions"], tab["probabilities"])):
    label = species.get(int(pred), str(pred))
    prob_str = "  ".join(f"{p:.2f}" for p in probs)
    print(f"  Query {i + 1}: predicted={label}  P(0|1|2)=[{prob_str}]")

# ---------------------------------------------------------------------------
# 5. Audio transcription
# ---------------------------------------------------------------------------

print("\n--- Audio Transcription ---")

sample_wav = pathlib.Path(__file__).parent / "sample.wav"
if not sample_wav.exists():
    print(f"  sample.wav not found at {sample_wav} — skipping transcription demo")
else:
    audio_b64 = base64.b64encode(sample_wav.read_bytes()).decode()
    audio_payload = {
        "model_type": "audio",
        "model_name": "transcriber",
        "audio_b64": audio_b64,
        "language": "en",
        "task": "transcribe",
    }
    r = requests.post(f"{BASE}/transcriber/predict", json=audio_payload)
    r.raise_for_status()
    aud = r.json()
    print(f"  Text     : {aud['text'].strip()}")
    print(f"  Language : {aud['language']}")
    if aud.get("duration"):
        print(f"  Duration : {aud['duration']:.2f}s")

# ---------------------------------------------------------------------------
# 6. Shut down
# ---------------------------------------------------------------------------

print("\nShutting down Ray Serve...")
server.shutdown()
print("Done.")
