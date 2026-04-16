"""TabPFN integration tests — require a real token and a live tabpfn install.

These tests call the real TabPFN library: no mocks, no stubs.
They exercise the full load() → predict() path against the live in-context
inference engine, gated behind two conditions:

  1. TABPFN_TOKEN env var is set (or a cached token exists under ~/.cache/tabpfn
     or ~/.tabpfn/token).
  2. tabpfn is importable (i.e. sheaf-serve[tabular] is installed).

Run explicitly:
    TABPFN_TOKEN=<token> uv run pytest tests/test_tabpfn_integration.py -v -s

Skipped automatically in CI unless both conditions are met.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Opt-in gate: skip unless tabpfn is installed and a token is available
# ---------------------------------------------------------------------------

_NO_TABPFN = False
try:
    import tabpfn  # noqa: F401
except ImportError:
    _NO_TABPFN = True


def _token_available() -> bool:
    """True if a token can be resolved by the same logic as TabPFNBackend.load()."""
    if os.environ.get("TABPFN_TOKEN"):
        return True
    import pathlib

    for path in (
        pathlib.Path.home() / ".cache" / "tabpfn" / "auth_token",
        pathlib.Path.home() / ".tabpfn" / "token",
    ):
        if path.exists() and path.read_text().strip():
            return True
    # Fall back to tabpfn's own get_cached_token if importable.
    try:
        from tabpfn.browser_auth import (
            get_cached_token,  # ty: ignore[unresolved-import]
        )

        return bool(get_cached_token())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    _NO_TABPFN or not _token_available(),
    reason="tabpfn not installed or no TABPFN_TOKEN available",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend():
    """Load a real TabPFNBackend once for the whole module."""
    from sheaf.backends.tabpfn import TabPFNBackend

    b = TabPFNBackend(device="cpu", n_estimators=2)
    b.load()
    return b


# ---------------------------------------------------------------------------
# Small synthetic datasets — intentionally trivially separable so that
# predictions are deterministic enough to assert on structure (not exact values).
# ---------------------------------------------------------------------------

# Binary classification: two clearly separated clusters.
_CLF_CONTEXT_X = [
    [0.0, 0.0],
    [0.1, 0.1],
    [0.2, 0.0],
    [10.0, 10.0],
    [10.1, 9.9],
    [9.9, 10.2],
]
_CLF_CONTEXT_Y = [0, 0, 0, 1, 1, 1]
_CLF_QUERY_X = [[0.05, 0.05], [10.0, 10.0]]  # should predict 0, 1 respectively

# Regression: y ≈ x₀ + x₁.
_REG_CONTEXT_X = [
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [4.0, 4.0],
    [5.0, 5.0],
]
_REG_CONTEXT_Y = [2.0, 4.0, 6.0, 8.0, 10.0]
_REG_QUERY_X = [[3.0, 3.0], [6.0, 6.0]]  # y ≈ 6, 12 respectively

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classification_predict_returns_response(backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.tabular import TabularRequest, TabularResponse

    req = TabularRequest(
        model_name="tabpfn",
        context_X=_CLF_CONTEXT_X,
        context_y=_CLF_CONTEXT_Y,
        query_X=_CLF_QUERY_X,
    )
    resp = backend.predict(req)

    assert isinstance(resp, TabularResponse)
    assert resp.task == "classification"
    assert resp.n_context == len(_CLF_CONTEXT_X)
    assert resp.n_query == len(_CLF_QUERY_X)
    assert len(resp.predictions) == len(_CLF_QUERY_X)
    assert resp.probabilities is None  # default output_mode="predictions"


def test_classification_predictions_are_valid_classes(backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.tabular import TabularRequest

    req = TabularRequest(
        model_name="tabpfn",
        context_X=_CLF_CONTEXT_X,
        context_y=_CLF_CONTEXT_Y,
        query_X=_CLF_QUERY_X,
    )
    resp = backend.predict(req)

    assert resp.classes is not None
    valid = set(resp.classes)
    for pred in resp.predictions:
        assert pred in valid, f"prediction {pred!r} not in classes {valid}"


def test_classification_separable_clusters_predict_correctly(backend) -> None:  # type: ignore[no-untyped-def]
    """Two well-separated clusters should be classified correctly."""
    from sheaf.api.tabular import TabularRequest

    req = TabularRequest(
        model_name="tabpfn",
        context_X=_CLF_CONTEXT_X,
        context_y=_CLF_CONTEXT_Y,
        query_X=_CLF_QUERY_X,
    )
    resp = backend.predict(req)

    # [0.05, 0.05] is near class-0 cluster; [10.0, 10.0] is in class-1 cluster.
    assert resp.predictions[0] == 0
    assert resp.predictions[1] == 1


def test_classification_probabilities(backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.tabular import TabularRequest

    req = TabularRequest(
        model_name="tabpfn",
        context_X=_CLF_CONTEXT_X,
        context_y=_CLF_CONTEXT_Y,
        query_X=_CLF_QUERY_X,
        output_mode="probabilities",
    )
    resp = backend.predict(req)

    assert resp.probabilities is not None
    assert len(resp.probabilities) == len(_CLF_QUERY_X)
    n_classes = len(resp.classes or [])
    for row in resp.probabilities:
        assert len(row) == n_classes
        assert abs(sum(row) - 1.0) < 1e-4, f"probabilities don't sum to 1: {row}"
        assert all(0.0 <= p <= 1.0 for p in row), f"probability out of [0,1]: {row}"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_regression_predict_returns_response(backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.tabular import TabularRequest, TabularResponse

    req = TabularRequest(
        model_name="tabpfn",
        context_X=_REG_CONTEXT_X,
        context_y=_REG_CONTEXT_Y,
        query_X=_REG_QUERY_X,
        task="regression",
    )
    resp = backend.predict(req)

    assert isinstance(resp, TabularResponse)
    assert resp.task == "regression"
    assert resp.n_context == len(_REG_CONTEXT_X)
    assert resp.n_query == len(_REG_QUERY_X)
    assert len(resp.predictions) == len(_REG_QUERY_X)
    assert resp.quantiles is None  # default output_mode="predictions"


def test_regression_predictions_are_finite_floats(backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.tabular import TabularRequest

    req = TabularRequest(
        model_name="tabpfn",
        context_X=_REG_CONTEXT_X,
        context_y=_REG_CONTEXT_Y,
        query_X=_REG_QUERY_X,
        task="regression",
    )
    resp = backend.predict(req)

    for pred in resp.predictions:
        assert np.isfinite(pred), f"non-finite prediction: {pred}"


def test_regression_quantiles(backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.tabular import TabularRequest

    ql = [0.1, 0.5, 0.9]
    req = TabularRequest(
        model_name="tabpfn",
        context_X=_REG_CONTEXT_X,
        context_y=_REG_CONTEXT_Y,
        query_X=_REG_QUERY_X,
        task="regression",
        output_mode="quantiles",
        quantile_levels=ql,
    )
    resp = backend.predict(req)

    assert resp.quantiles is not None
    assert set(resp.quantiles.keys()) == {str(q) for q in ql}
    for key, vals in resp.quantiles.items():
        assert len(vals) == len(_REG_QUERY_X), f"quantile {key}: wrong length"
        assert all(np.isfinite(v) for v in vals), f"quantile {key}: non-finite values"

    # Monotonicity: q10 ≤ q50 ≤ q90 for each query row.
    q10 = resp.quantiles["0.1"]
    q50 = resp.quantiles["0.5"]
    q90 = resp.quantiles["0.9"]
    for i, (a, b, c) in enumerate(zip(q10, q50, q90)):
        assert a <= b + 1e-4, f"row {i}: q10={a} > q50={b}"
        assert b <= c + 1e-4, f"row {i}: q50={b} > q90={c}"


# ---------------------------------------------------------------------------
# batch_predict
# ---------------------------------------------------------------------------


def test_batch_predict_runs_requests_independently(backend) -> None:  # type: ignore[no-untyped-def]
    """batch_predict with two different requests returns two independent responses."""
    from sheaf.api.tabular import TabularRequest, TabularResponse

    req_clf = TabularRequest(
        model_name="tabpfn",
        context_X=_CLF_CONTEXT_X,
        context_y=_CLF_CONTEXT_Y,
        query_X=_CLF_QUERY_X,
    )
    req_reg = TabularRequest(
        model_name="tabpfn",
        context_X=_REG_CONTEXT_X,
        context_y=_REG_CONTEXT_Y,
        query_X=_REG_QUERY_X,
        task="regression",
    )
    responses = backend.batch_predict([req_clf, req_reg])

    assert len(responses) == 2
    assert all(isinstance(r, TabularResponse) for r in responses)
    assert responses[0].task == "classification"
    assert responses[1].task == "regression"
