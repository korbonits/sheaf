"""Tests for TabPFNBackend — fully mocked, no real tabpfn install required.

Covers:
  - load() raises OSError when TABPFN_TOKEN is absent (container-friendliness)
  - load() succeeds when TABPFN_TOKEN is set (no browser flow triggered)
  - predict() end-to-end for classification and regression
  - output_mode="probabilities" (classification) and "quantiles" (regression)
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.tabular import TabularRequest, TabularResponse
from sheaf.backends.tabpfn import TabPFNBackend

# ---------------------------------------------------------------------------
# Mock tabpfn module fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tabpfn() -> ModuleType:
    """Fake tabpfn module with stub Classifier and Regressor.

    The stubs mimic the sklearn-style API: fit(), predict(), predict_proba(),
    and classes_ that TabPFNBackend depends on.
    """
    mod = ModuleType("tabpfn")

    # --- Classifier stub ---
    clf_instance = MagicMock()
    clf_instance.predict.return_value = np.array([0, 1])
    clf_instance.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    clf_instance.classes_ = np.array([0, 1])
    MockClassifier = MagicMock(return_value=clf_instance)

    # --- Regressor stub ---
    reg_instance = MagicMock()
    reg_instance.predict.return_value = np.array([1.5, 2.5])
    reg_instance.predict.side_effect = lambda X, output_type="mean", **kw: (
        np.array([1.5, 2.5])
        if output_type == "mean"
        else np.array([[1.0, 1.5], [2.0, 2.5], [1.8, 2.3]])  # 3 quantiles × 2 rows
    )
    MockRegressor = MagicMock(return_value=reg_instance)

    mod.TabPFNClassifier = MockClassifier
    mod.TabPFNRegressor = MockRegressor
    return mod


@pytest.fixture
def loaded_backend(mock_tabpfn: ModuleType) -> TabPFNBackend:
    """TabPFNBackend with load() already called (TABPFN_TOKEN set, tabpfn mocked)."""
    backend = TabPFNBackend()
    with (
        patch.dict(os.environ, {"TABPFN_TOKEN": "test-token-does-not-call-browser"}),
        patch.dict(sys.modules, {"tabpfn": mock_tabpfn}),
    ):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# Auth / load
# ---------------------------------------------------------------------------


def test_load_raises_without_token() -> None:
    """load() must raise OSError when TABPFN_TOKEN is absent.

    This is the container-friendliness guarantee: no token → clear error,
    no browser pop-up attempted.
    """
    backend = TabPFNBackend()
    env_without_token = {k: v for k, v in os.environ.items() if k != "TABPFN_TOKEN"}
    with (
        patch.dict(os.environ, env_without_token, clear=True),
        pytest.raises(OSError, match="TABPFN_TOKEN"),
    ):
        backend.load()


def test_load_with_token_does_not_raise(mock_tabpfn: ModuleType) -> None:
    """load() succeeds when TABPFN_TOKEN is set — no browser flow triggered."""
    backend = TabPFNBackend()
    with (
        patch.dict(os.environ, {"TABPFN_TOKEN": "test-token"}),
        patch.dict(sys.modules, {"tabpfn": mock_tabpfn}),
    ):
        backend.load()  # must not raise
    assert backend._classifier_cls is not None
    assert backend._regressor_cls is not None


def test_load_raises_on_missing_tabpfn_package() -> None:
    """load() raises ImportError when tabpfn is not installed."""
    backend = TabPFNBackend()
    # Remove tabpfn from sys.modules so the import inside load() fails
    modules_without_tabpfn = {k: v for k, v in sys.modules.items() if k != "tabpfn"}
    with (
        patch.dict(os.environ, {"TABPFN_TOKEN": "test-token"}),
        patch.dict(sys.modules, modules_without_tabpfn, clear=True),
        patch("builtins.__import__", side_effect=_raise_on_tabpfn),
        pytest.raises(ImportError, match="sheaf-serve\\[tabular\\]"),
    ):
        backend.load()


def _raise_on_tabpfn(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
    if name == "tabpfn":
        raise ModuleNotFoundError("No module named 'tabpfn'")
    return __import__(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_predict_classification(loaded_backend: TabPFNBackend) -> None:
    req = TabularRequest(
        model_name="tabpfn",
        context_X=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        context_y=[0, 1, 0],
        query_X=[[2.0, 3.0], [4.0, 5.0]],
    )
    resp = loaded_backend.predict(req)
    assert isinstance(resp, TabularResponse)
    assert resp.task == "classification"
    assert resp.predictions == [0, 1]
    assert resp.probabilities is None
    assert resp.classes == [0, 1]
    assert resp.n_context == 3
    assert resp.n_query == 2


def test_predict_classification_with_probabilities(
    loaded_backend: TabPFNBackend,
) -> None:
    req = TabularRequest(
        model_name="tabpfn",
        context_X=[[1.0, 2.0], [3.0, 4.0]],
        context_y=[0, 1],
        query_X=[[2.0, 3.0], [4.0, 5.0]],
        output_mode="probabilities",
    )
    resp = loaded_backend.predict(req)
    assert resp.probabilities is not None
    assert len(resp.probabilities) == 2  # one row per query
    assert len(resp.probabilities[0]) == 2  # one prob per class


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_predict_regression(loaded_backend: TabPFNBackend) -> None:
    req = TabularRequest(
        model_name="tabpfn",
        context_X=[[1.0, 2.0], [3.0, 4.0]],
        context_y=[1.2, 3.4],
        query_X=[[2.0, 3.0], [4.0, 5.0]],
        task="regression",
    )
    resp = loaded_backend.predict(req)
    assert isinstance(resp, TabularResponse)
    assert resp.task == "regression"
    assert resp.predictions == [1.5, 2.5]
    assert resp.quantiles is None


def test_predict_regression_with_quantiles(loaded_backend: TabPFNBackend) -> None:
    req = TabularRequest(
        model_name="tabpfn",
        context_X=[[1.0, 2.0], [3.0, 4.0]],
        context_y=[1.2, 3.4],
        query_X=[[2.0, 3.0], [4.0, 5.0]],
        task="regression",
        output_mode="quantiles",
        quantile_levels=[0.1, 0.5, 0.9],
    )
    resp = loaded_backend.predict(req)
    assert resp.quantiles is not None
    assert set(resp.quantiles.keys()) == {"0.1", "0.5", "0.9"}
    assert len(resp.quantiles["0.1"]) == 2  # one value per query row
