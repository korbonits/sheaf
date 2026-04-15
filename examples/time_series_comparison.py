"""Side-by-side comparison: Chronos-Bolt vs TimesFM on the same request.

Demonstrates that the same TimeSeriesRequest contract works across backends.

Usage:
    pip install "sheaf-serve[time-series]"
    python examples/time_series_comparison.py
"""

from sheaf.api.time_series import Frequency, OutputMode, TimeSeriesRequest
from sheaf.backends.chronos import Chronos2Backend
from sheaf.backends.timesfm import TimesFMBackend

# Hourly electricity demand (kWh), last 24 hours
history = [
    312, 298, 275, 260, 255, 263, 285, 320,
    368, 402, 421, 435, 442, 438, 430, 425,
    418, 410, 398, 385, 372, 358, 342, 328,
]

req = TimeSeriesRequest(
    model_name="comparison",
    history=history,
    horizon=12,
    frequency=Frequency.HOURLY,
    output_mode=OutputMode.QUANTILES,
    quantile_levels=[0.1, 0.5, 0.9],
)

# --- Load both backends ---

print("Loading Chronos-Bolt...")
chronos = Chronos2Backend(model_id="amazon/chronos-bolt-tiny", device_map="cpu")
chronos.load()

print("Loading TimesFM...")
timesfm = TimesFMBackend(model_id="google/timesfm-1.0-200m-pytorch", backend="cpu")
timesfm.load()

print("\nForecasting next 12 hours...\n")

chronos_resp = chronos.predict(req)
timesfm_resp = timesfm.predict(req)

# --- Print side-by-side ---

print(f"{'Hour':>5}  {'Chronos P10':>12}  {'Chronos P50':>12}  {'Chronos P90':>12}  "
      f"{'TimesFM P10':>12}  {'TimesFM P50':>12}  {'TimesFM P90':>12}")
print("-" * 91)

c_p10 = chronos_resp.quantiles["0.1"]
c_p50 = chronos_resp.quantiles["0.5"]
c_p90 = chronos_resp.quantiles["0.9"]

t_p10 = timesfm_resp.quantiles["0.1"]
t_p50 = timesfm_resp.quantiles["0.5"]
t_p90 = timesfm_resp.quantiles["0.9"]

for i in range(req.horizon):
    print(
        f"{i+1:>5}  {c_p10[i]:>12.1f}  {c_p50[i]:>12.1f}  {c_p90[i]:>12.1f}  "
        f"{t_p10[i]:>12.1f}  {t_p50[i]:>12.1f}  {t_p90[i]:>12.1f}"
    )
