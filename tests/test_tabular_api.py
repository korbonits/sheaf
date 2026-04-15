"""Tests for tabular API contract."""

import pytest
from pydantic import ValidationError

from sheaf.api.tabular import TabularRequest


def _base_request(**kwargs) -> dict:
    return {
        "model_name": "tabpfn",
        "context_X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "context_y": [0, 1, 0],
        "query_X": [[2.0, 3.0], [4.0, 5.0]],
        **kwargs,
    }


def test_classification_default():
    req = TabularRequest(**_base_request())
    assert req.task == "classification"
    assert req.output_mode == "predictions"
    assert len(req.context_X) == 3


def test_regression_with_quantiles():
    req = TabularRequest(
        **_base_request(
            context_y=[1.2, 3.4, 2.1],
            task="regression",
            output_mode="quantiles",
            quantile_levels=[0.1, 0.5, 0.9],
        )
    )
    assert req.task == "regression"
    assert req.quantile_levels == [0.1, 0.5, 0.9]


def test_rejects_mismatched_context_shapes():
    with pytest.raises(ValidationError, match="context_X"):
        TabularRequest(
            **_base_request(
                context_y=[0, 1],  # 2 labels but 3 context rows
            )
        )


def test_rejects_mismatched_feature_counts():
    with pytest.raises(ValidationError, match="features"):
        TabularRequest(
            **_base_request(
                query_X=[[1.0, 2.0, 3.0]],  # 3 features vs 2 in context
            )
        )


def test_rejects_probabilities_for_regression():
    with pytest.raises(ValidationError, match="probabilities"):
        TabularRequest(
            **_base_request(
                context_y=[1.0, 2.0, 3.0],
                task="regression",
                output_mode="probabilities",
            )
        )


def test_rejects_quantiles_for_classification():
    with pytest.raises(ValidationError, match="quantiles"):
        TabularRequest(
            **_base_request(
                task="classification",
                output_mode="quantiles",
            )
        )
