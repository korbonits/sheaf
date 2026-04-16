"""Tests for TabPFNBackend — fully mocked, no real tabpfn install required.

Covers:
  - load() raises OSError when no token is available anywhere (container-friendliness)
  - load() succeeds when get_cached_token() returns a token (env var, file cache, etc.)
  - TABPFN_NO_BROWSER is set by load() to prevent browser pop-ups
  - TabPFNLicenseError at fit() time is re-raised as OSError
  - predict() end-to-end for classification and regression
  - output_mode="probabilities" (classification) and "quantiles" (regression)
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.tabular import TabularRequest, TabularResponse
from sheaf.backends.tabpfn import TabPFNBackend

# ---------------------------------------------------------------------------
# Mock tabpfn module fixtures
# ---------------------------------------------------------------------------


def _make_tabpfn_mocks() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Build fake tabpfn, tabpfn.browser_auth, and tabpfn.errors modules.

    Returns (tabpfn_mod, browser_auth_mod, errors_mod).
    """
    # --- tabpfn.errors ---
    errors_mod = ModuleType("tabpfn.errors")

    class TabPFNLicenseError(Exception):
        pass

    errors_mod.TabPFNLicenseError = TabPFNLicenseError  # type: ignore[attr-defined]

    # --- tabpfn.browser_auth ---
    browser_auth_mod = ModuleType("tabpfn.browser_auth")
    browser_auth_mod.get_cached_token = MagicMock(return_value="test-token")  # type: ignore[attr-defined]

    # --- tabpfn ---
    tabpfn_mod = ModuleType("tabpfn")

    clf_instance = MagicMock()
    clf_instance.predict.return_value = np.array([0, 1])
    clf_instance.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    clf_instance.classes_ = np.array([0, 1])
    tabpfn_mod.TabPFNClassifier = MagicMock(return_value=clf_instance)  # type: ignore[attr-defined]

    reg_instance = MagicMock()
    reg_instance.predict.side_effect = lambda X, output_type="mean", **kw: (
        np.array([1.5, 2.5])
        if output_type == "mean"
        else np.array([[1.0, 1.5], [2.0, 2.5], [1.8, 2.3]])  # 3 quantiles × 2 rows
    )
    tabpfn_mod.TabPFNRegressor = MagicMock(return_value=reg_instance)  # type: ignore[attr-defined]

    return tabpfn_mod, browser_auth_mod, errors_mod


@pytest.fixture
def mock_tabpfn_modules() -> dict[str, ModuleType]:
    """sys.modules patch dict covering tabpfn, tabpfn.browser_auth, tabpfn.errors."""
    tabpfn_mod, browser_auth_mod, errors_mod = _make_tabpfn_mocks()
    return {
        "tabpfn": tabpfn_mod,
        "tabpfn.browser_auth": browser_auth_mod,
        "tabpfn.errors": errors_mod,
    }


@pytest.fixture
def loaded_backend(mock_tabpfn_modules: dict[str, ModuleType]) -> TabPFNBackend:
    """TabPFNBackend with load() already called (token mocked, tabpfn mocked)."""
    backend = TabPFNBackend()
    with patch.dict(sys.modules, mock_tabpfn_modules):
        backend.load()
    return backend


# ---------------------------------------------------------------------------
# Auth / load
# ---------------------------------------------------------------------------


def test_load_raises_without_token(
    mock_tabpfn_modules: dict[str, ModuleType],
) -> None:
    """load() must raise OSError when no token is resolvable from any source.

    This is the container-friendliness guarantee: no token → clear error,
    no browser pop-up attempted.
    """
    mock_tabpfn_modules["tabpfn.browser_auth"].get_cached_token.return_value = None  # type: ignore[attr-defined]
    backend = TabPFNBackend()
    with (
        patch.dict(sys.modules, mock_tabpfn_modules),
        pytest.raises(OSError, match="No TabPFN token found"),
    ):
        backend.load()


def test_load_with_token_does_not_raise(
    mock_tabpfn_modules: dict[str, ModuleType],
) -> None:
    """load() succeeds when get_cached_token() returns a token."""
    backend = TabPFNBackend()
    with patch.dict(sys.modules, mock_tabpfn_modules):
        backend.load()  # must not raise
    assert backend._classifier_cls is not None
    assert backend._regressor_cls is not None


def test_load_sets_no_browser_env(
    mock_tabpfn_modules: dict[str, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load() sets TABPFN_NO_BROWSER so headless environments never get a pop-up."""
    monkeypatch.delenv("TABPFN_NO_BROWSER", raising=False)
    backend = TabPFNBackend()
    with patch.dict(sys.modules, mock_tabpfn_modules):
        backend.load()
    import os

    assert os.environ.get("TABPFN_NO_BROWSER") == "1"


def test_load_raises_on_missing_tabpfn_package() -> None:
    """load() raises ImportError when tabpfn is not installed."""
    backend = TabPFNBackend()
    modules_without_tabpfn = {
        k: v for k, v in sys.modules.items() if not k.startswith("tabpfn")
    }
    with (
        patch.dict(sys.modules, modules_without_tabpfn, clear=True),
        patch("builtins.__import__", side_effect=_raise_on_tabpfn),
        pytest.raises(ImportError, match="sheaf-serve\\[tabular\\]"),
    ):
        backend.load()


def _raise_on_tabpfn(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
    if name == "tabpfn":
        raise ModuleNotFoundError("No module named 'tabpfn'")
    return __import__(name, *args, **kwargs)


def test_predict_raises_on_license_error_at_fit(
    mock_tabpfn_modules: dict[str, ModuleType],
) -> None:
    """predict() re-raises TabPFNLicenseError from fit() as OSError."""
    TabPFNLicenseError = mock_tabpfn_modules["tabpfn.errors"].TabPFNLicenseError  # type: ignore[attr-defined]

    backend = TabPFNBackend()
    # Make the classifier stub raise TabPFNLicenseError on fit()
    clf_instance = mock_tabpfn_modules["tabpfn"].TabPFNClassifier.return_value  # type: ignore[attr-defined]
    clf_instance.fit.side_effect = TabPFNLicenseError("token expired")

    req = TabularRequest(
        model_name="tabpfn",
        context_X=[[1.0, 2.0], [3.0, 4.0]],
        context_y=[0, 1],
        query_X=[[2.0, 3.0]],
    )
    # Both load() and predict() must run inside the patch so _run_classification
    # imports the same TabPFNLicenseError class that fit() raises.
    with (
        patch.dict(sys.modules, mock_tabpfn_modules),
        pytest.raises(OSError, match="license error"),
    ):
        backend.load()
        backend.predict(req)


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
