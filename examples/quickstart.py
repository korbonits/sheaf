"""Sheaf quickstart — time series forecasting with Chronos-Bolt.

Downloads amazon/chronos-bolt-tiny (~80MB) on first run.
Runs on CPU, no GPU required.

Usage:
    pip install "sheaf-serve[time-series]"
    python examples/quickstart.py
"""

from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest
from sheaf.backends.chronos import Chronos2Backend

# --- 1. Load the model ---

print("Loading amazon/chronos-bolt-tiny...")
backend = Chronos2Backend(
    model_id="amazon/chronos-bolt-tiny",
    device_map="cpu",
    torch_dtype="float32",
)
backend.load()
print("Ready.\n")

# --- 2. Build a request ---

# Hourly electricity demand (kWh), last 24 hours
history = [
    312, 298, 275, 260, 255, 263, 285, 320,
    368, 402, 421, 435, 442, 438, 430, 425,
    418, 410, 398, 385, 372, 358, 342, 328,
]

req = TimeSeriesRequest(
    model_name="chronos-bolt-tiny",
    history=history,
    horizon=12,
    frequency=Frequency.HOURLY,
    output_mode=OutputMode.QUANTILES,
    quantile_levels=[0.1, 0.5, 0.9],
)

# --- 3. Run inference ---

response = backend.predict(req)

# --- 4. Print results ---

print(f"Forecast: next {response.horizon} hours\n")
print(f"{'Hour':>5}  {'P10':>7}  {'Median':>7}  {'P90':>7}")
print("-" * 35)

p10 = response.quantiles["0.1"]
p50 = response.quantiles["0.5"]
p90 = response.quantiles["0.9"]

for i in range(response.horizon):
    print(f"{i+1:>5}  {p10[i]:>7.1f}  {p50[i]:>7.1f}  {p90[i]:>7.1f}")

# --- 5. Batch prediction ---

print("\n--- Batch prediction (3 series) ---\n")

series = [
    [100, 110, 108, 115, 120, 118, 125, 130],
    [500, 480, 460, 450, 445, 440, 435, 430],
    [10, 12, 11, 13, 15, 14, 16, 17],
]

requests = [
    TimeSeriesRequest(
        model_name="chronos-bolt-tiny",
        history=h,
        horizon=6,
        frequency=Frequency.DAILY,
        output_mode=OutputMode.MEAN,
    )
    for h in series
]

responses = backend.batch_predict(requests)

for i, resp in enumerate(responses):
    print(f"Series {i+1}: {[round(x, 1) for x in resp.mean]}")
