"""Sheaf quickstart on Modal — GPU inference without a local GPU.

Runs three heterogeneous sheaf backends on Modal cloud compute:
  - ChronosForecaster  : amazon/chronos-bolt-tiny on GPU (T4)
  - TabularClassifier  : TabPFN v2 on CPU
  - Transcriber        : openai/whisper-tiny on GPU (T4)

Prerequisites:
    pip install modal
    modal setup              # authenticate once

    # TabPFN token (https://ux.priorlabs.ai) — store as a Modal secret:
    modal secret create tabpfn-token TABPFN_TOKEN=<your-token>

Run:
    modal run examples/quickstart_modal.py

Each class loads its model once when its container starts (@modal.enter),
then serves requests until the container is recycled.  min_containers=1
keeps one warm instance alive so the demo doesn't cold-start on each call.
"""

from __future__ import annotations

import base64
import os
import pathlib

import modal

app = modal.App("sheaf-quickstart")

# ---------------------------------------------------------------------------
# Images — one per model family to keep layers independent
# ---------------------------------------------------------------------------

# sheaf-serve is not yet published with these backends; install local source.
# add_local_python_source mounts src/sheaf into the container at build time.
_sheaf_deps = ["pydantic>=2.0.0", "numpy>=1.24.0"]

# Chronos only (skip TimesFM/JAX to keep the image lean for this demo)
_ts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_sheaf_deps, "chronos-forecasting>=1.0.0", "torch>=2.1,<2.5")
    .add_local_python_source("sheaf")
)

# TabPFN — CPU-only, no torch extras needed beyond what tabpfn pulls in
_tab_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_sheaf_deps, "tabpfn>=2.0.0")
    .add_local_python_source("sheaf")
)

# Whisper — needs torch; WAV inputs are decoded in pure Python (no ffmpeg needed)
_audio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_sheaf_deps, "openai-whisper>=20240930", "torch>=2.1,<2.5")
    .add_local_python_source("sheaf")
)

# ---------------------------------------------------------------------------
# Time series: Chronos-Bolt
# ---------------------------------------------------------------------------


_hf_secret = modal.Secret.from_name("huggingface")


@app.cls(image=_ts_image, gpu="T4", min_containers=1, secrets=[_hf_secret])
class ChronosForecaster:
    @modal.enter()
    def load(self) -> None:
        from sheaf.backends.chronos import Chronos2Backend

        self.backend = Chronos2Backend(
            model_id="amazon/chronos-bolt-tiny",
            device_map="cuda",
            torch_dtype="float32",
        )
        self.backend.load()

    @modal.method()
    def forecast(
        self,
        history: list[float],
        horizon: int = 6,
        quantile_levels: list[float] | None = None,
    ) -> dict:
        from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest

        req = TimeSeriesRequest(
            model_name="chronos-bolt-tiny",
            history=history,
            horizon=horizon,
            frequency=Frequency.HOURLY,
            output_mode=OutputMode.QUANTILES,
            quantile_levels=quantile_levels or [0.1, 0.5, 0.9],
        )
        return self.backend.predict(req).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tabular: TabPFN v2
# ---------------------------------------------------------------------------

# Pass TABPFN_TOKEN from local env, or an empty string if not set.
# TabPFN.load() will raise OSError if the token is missing/invalid —
# the local entrypoint skips the classifier in that case.
_tabpfn_secret = modal.Secret.from_dict(
    {"TABPFN_TOKEN": os.environ.get("TABPFN_TOKEN", "")}
)


@app.cls(image=_tab_image, secrets=[_tabpfn_secret, _hf_secret], min_containers=1)
class TabularClassifier:
    @modal.enter()
    def load(self) -> None:
        from sheaf.backends.tabpfn import TabPFNBackend

        self.backend = TabPFNBackend()
        self.backend.load()

    @modal.method()
    def classify(
        self,
        context_X: list[list[float]],
        context_y: list[float | int],
        query_X: list[list[float]],
    ) -> dict:
        from sheaf.api.tabular import TabularRequest

        req = TabularRequest(
            model_name="tabpfn",
            context_X=context_X,
            context_y=context_y,
            query_X=query_X,
            task="classification",
            output_mode="probabilities",
        )
        return self.backend.predict(req).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Audio: Whisper-tiny
# ---------------------------------------------------------------------------


@app.cls(image=_audio_image, gpu="T4", min_containers=1, secrets=[_hf_secret])
class Transcriber:
    @modal.enter()
    def load(self) -> None:
        from sheaf.backends.whisper import WhisperBackend

        self.backend = WhisperBackend(model_size="tiny", device="cuda")
        self.backend.load()

    @modal.method()
    def transcribe(self, audio_b64: str, language: str = "en") -> dict:
        from sheaf.api.audio import AudioRequest

        req = AudioRequest(
            model_name="whisper-tiny",
            audio_b64=audio_b64,
            language=language,
            task="transcribe",
        )
        return self.backend.predict(req).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates all three and prints results
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    # --- Time series forecast ---
    print("\n--- Time Series Forecast (Chronos-Bolt on GPU T4) ---")

    history = [312, 298, 275, 260, 255, 263, 285, 320, 368, 402, 421, 435]
    ts = ChronosForecaster().forecast.remote(history=history, horizon=6)

    print(f"{'Hour':>5}  {'P10':>7}  {'Median':>7}  {'P90':>7}")
    print("-" * 35)
    p10 = ts["quantiles"]["0.1"]
    p50 = ts["quantiles"]["0.5"]
    p90 = ts["quantiles"]["0.9"]
    for i in range(ts["horizon"]):
        print(f"{i + 1:>5}  {p10[i]:>7.1f}  {p50[i]:>7.1f}  {p90[i]:>7.1f}")

    # --- Tabular classification ---
    print("\n--- Tabular Classification — Iris (TabPFN on CPU) ---")

    # Iris: sepal_length, sepal_width, petal_length (3 features)
    context_X = [
        [5.1, 3.5, 1.4],  # setosa
        [4.9, 3.0, 1.4],  # setosa
        [7.0, 3.2, 4.7],  # versicolor
        [6.4, 3.2, 4.5],  # versicolor
        [6.3, 3.3, 6.0],  # virginica
        [5.8, 2.7, 5.1],  # virginica
    ]
    context_y = [0, 0, 1, 1, 2, 2]
    query_X = [
        [5.0, 3.4, 1.5],  # likely setosa
        [6.1, 2.8, 4.7],  # likely versicolor
    ]

    tab = TabularClassifier().classify.remote(
        context_X=context_X,
        context_y=context_y,
        query_X=query_X,
    )

    species = {0: "setosa", 1: "versicolor", 2: "virginica"}
    for i, (pred, probs) in enumerate(zip(tab["predictions"], tab["probabilities"])):
        label = species.get(int(pred), str(pred))
        prob_str = "  ".join(f"{p:.2f}" for p in probs)
        print(f"  Query {i + 1}: predicted={label}  P(0|1|2)=[{prob_str}]")

    # --- Audio transcription ---
    print("\n--- Audio Transcription (Whisper-tiny on GPU T4) ---")

    sample_wav = pathlib.Path(__file__).parent / "sample.wav"
    if not sample_wav.exists():
        print(f"  sample.wav not found at {sample_wav} — skipping")
    else:
        audio_b64 = base64.b64encode(sample_wav.read_bytes()).decode()
        aud = Transcriber().transcribe.remote(audio_b64=audio_b64, language="en")
        print(f"  Text     : {aud['text'].strip()}")
        print(f"  Language : {aud['language']}")
        if aud.get("duration"):
            print(f"  Duration : {aud['duration']:.2f}s")

    print("\nDone.")
