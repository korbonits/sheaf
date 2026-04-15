"""API contract for tabular foundation models (TabPFN, etc.)."""

from typing import Literal

from pydantic import Field, model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class TabularRequest(BaseRequest):
    """Request contract for tabular foundation models.

    TabPFN is an in-context learner: context_X/context_y are the "training"
    examples passed at inference time. query_X contains the rows to predict.
    No separate training step — everything happens in a single forward pass.

    Args:
        context_X: Feature matrix for in-context examples, shape [n_context, n_features]
        context_y: Labels for in-context examples, shape [n_context]
        query_X: Feature rows to predict, shape [n_query, n_features]
        task: "classification" or "regression"
        feature_names: Optional column names — used for logging and debugging
        categorical_feature_indices: Indices of categorical columns
        output_mode: "predictions" for point predictions only,
                     "probabilities" for classification probabilities,
                     "quantiles" for regression quantile estimates
        quantile_levels: Quantile levels — only used when task=regression
                         and output_mode=quantiles
    """

    model_type: ModelType = ModelType.TABULAR

    context_X: list[list[float]]
    context_y: list[float | int]
    query_X: list[list[float]]
    task: Literal["classification", "regression"] = "classification"
    feature_names: list[str] | None = None
    categorical_feature_indices: list[int] | None = None
    output_mode: Literal["predictions", "probabilities", "quantiles"] = "predictions"
    quantile_levels: list[float] = Field(default=[0.1, 0.5, 0.9])

    @model_validator(mode="after")
    def validate_shapes(self) -> "TabularRequest":
        if len(self.context_X) != len(self.context_y):
            raise ValueError(
                f"context_X has {len(self.context_X)} rows but "
                f"context_y has {len(self.context_y)} elements."
            )
        if self.context_X and self.query_X:
            n_ctx_features = len(self.context_X[0])
            n_query_features = len(self.query_X[0])
            if n_ctx_features != n_query_features:
                raise ValueError(
                    f"context_X has {n_ctx_features} features but "
                    f"query_X has {n_query_features} features."
                )
        if self.output_mode == "probabilities" and self.task == "regression":
            raise ValueError(
                "output_mode='probabilities' is only valid for task='classification'."
            )
        if self.output_mode == "quantiles" and self.task == "classification":
            raise ValueError(
                "output_mode='quantiles' is only valid for task='regression'."
            )
        return self


class TabularResponse(BaseResponse):
    """Response contract for tabular foundation models."""

    model_type: ModelType = ModelType.TABULAR

    # Point predictions — always populated
    # Classification: predicted class labels
    # Regression: predicted values (mean)
    predictions: list[float | int]

    # Classification only: class probabilities, shape [n_query, n_classes]
    probabilities: list[list[float]] | None = None

    # Regression only: quantile estimates keyed by quantile level
    # e.g. {"0.1": [...], "0.5": [...], "0.9": [...]}
    quantiles: dict[str, list[float]] | None = None

    # Classification only: class labels corresponding to probability columns
    classes: list[int | str] | None = None

    task: str
    n_context: int
    n_query: int
