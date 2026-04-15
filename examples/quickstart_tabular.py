"""Sheaf quickstart — tabular prediction with TabPFN.

Requires TABPFN_TOKEN environment variable:
    export TABPFN_TOKEN="<your-token>"  # https://ux.priorlabs.ai

Usage:
    pip install "sheaf-serve[tabular]"
    TABPFN_TOKEN=<token> python examples/quickstart_tabular.py
"""

import numpy as np

from sheaf.api.tabular import TabularRequest
from sheaf.backends.tabpfn import TabPFNBackend

np.random.seed(42)

backend = TabPFNBackend(device="cpu", n_estimators=4)
backend.load()
print("TabPFN loaded.\n")

# --- Classification ---

print("=== Classification ===\n")

# Iris-like: predict species from petal/sepal measurements
context_X = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.3, 3.3, 4.7, 1.6],
    [5.8, 2.7, 4.1, 1.0],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [5.0, 3.6, 1.4, 0.2],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 3.0, 5.8, 2.2],
]
context_y = [0, 0, 1, 1, 2, 2, 0, 1, 2]

query_X = [
    [5.0, 3.4, 1.5, 0.2],  # expect: 0 (setosa)
    [6.0, 2.9, 4.5, 1.5],  # expect: 1 (versicolor)
    [6.8, 3.0, 5.5, 2.1],  # expect: 2 (virginica)
]

req = TabularRequest(
    model_name="tabpfn",
    context_X=context_X,
    context_y=context_y,
    query_X=query_X,
    task="classification",
    output_mode="probabilities",
    feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)

resp = backend.predict(req)

species = {0: "setosa", 1: "versicolor", 2: "virginica"}
header = (
    f"{'Query':>5}  {'Prediction':>12}  "
    f"{'P(setosa)':>10}  {'P(versicolor)':>14}  {'P(virginica)':>13}"
)
print(header)
print("-" * 65)
for i, (pred, proba) in enumerate(zip(resp.predictions, resp.probabilities)):
    print(
        f"{i + 1:>5}  {species[pred]:>12}  "
        f"{proba[0]:>10.3f}  {proba[1]:>14.3f}  {proba[2]:>13.3f}"
    )

# --- Regression ---

print("\n=== Regression ===\n")

# Predict house price from features
np.random.seed(0)
n_ctx = 30
size = np.random.uniform(50, 200, n_ctx)
bedrooms = np.random.randint(1, 5, n_ctx).astype(float)
age = np.random.uniform(0, 50, n_ctx)
price = size * 3000 + bedrooms * 20000 - age * 1000 + np.random.randn(n_ctx) * 10000

ctx_X = np.stack([size, bedrooms, age], axis=1).tolist()
ctx_y = price.tolist()

query = [[120.0, 3.0, 10.0], [80.0, 2.0, 30.0], [180.0, 4.0, 5.0]]

req = TabularRequest(
    model_name="tabpfn",
    context_X=ctx_X,
    context_y=ctx_y,
    query_X=query,
    task="regression",
    output_mode="quantiles",
    quantile_levels=[0.1, 0.5, 0.9],
    feature_names=["size_sqm", "bedrooms", "age_years"],
)

resp = backend.predict(req)

labels = ["120sqm/3bed/10yr", "80sqm/2bed/30yr", "180sqm/4bed/5yr"]
print(f"{'House':>20}  {'P10':>10}  {'Median':>10}  {'P90':>10}")
print("-" * 58)
for i, label in enumerate(labels):
    p10 = resp.quantiles["0.1"][i]
    p50 = resp.quantiles["0.5"][i]
    p90 = resp.quantiles["0.9"][i]
    print(f"{label:>20}  {p10:>10,.0f}  {p50:>10,.0f}  {p90:>10,.0f}")
