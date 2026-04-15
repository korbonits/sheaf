"""TabPFN backend for tabular foundation models.

Requires: pip install "sheaf-serve[tabular]"
Requires: TABPFN_TOKEN environment variable (one-time license acceptance)
          See https://ux.priorlabs.ai to obtain a token.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.tabular import TabularRequest, TabularResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("tabpfn")
class TabPFNBackend(ModelBackend):
    """ModelBackend for TabPFN v2 (classification and regression).

    TabPFN is an in-context learner: each request passes context examples
    (context_X, context_y) alongside query rows (query_X). There are no
    persistent model weights per dataset — the forward pass handles everything.

    Requires TABPFN_TOKEN to be set in the environment:
        export TABPFN_TOKEN="<your-api-key>"

    Args:
        device: "cpu", "cuda", "mps", or "auto"
        n_estimators: Number of ensemble estimators (more = slower, better)
        inference_precision: "auto", "autocast", or a torch dtype string
    """

    def __init__(
        self,
        device: str = "cpu",
        n_estimators: int = 4,
        inference_precision: str = "auto",
    ) -> None:
        self._device = device
        self._n_estimators = n_estimators
        self._inference_precision = inference_precision
        self._classifier_cls: Any = None
        self._regressor_cls: Any = None

    @property
    def model_type(self) -> str:
        return ModelType.TABULAR

    def load(self) -> None:
        if not os.environ.get("TABPFN_TOKEN"):
            raise OSError(
                "TABPFN_TOKEN is not set. Obtain a token at https://ux.priorlabs.ai "
                "and set it with: export TABPFN_TOKEN='<your-token>'"
            )
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:
            raise ImportError(
                "tabpfn is required for the TabPFN backend. "
                "Install it with: pip install 'sheaf-serve[tabular]'"
            ) from e

        self._classifier_cls = TabPFNClassifier
        self._regressor_cls = TabPFNRegressor

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, TabularRequest):
            raise TypeError(f"Expected TabularRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        # TabPFN's serving bottleneck is context table memory, not query batching.
        # Each request may have a different context table, so we run independently.
        # Future: cache context tables and batch query rows against the same context.
        return [self.predict(r) for r in requests]

    def _run(self, request: TabularRequest) -> TabularResponse:
        if self._classifier_cls is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        X_ctx = np.array(request.context_X, dtype=np.float32)
        y_ctx = np.array(request.context_y)
        X_query = np.array(request.query_X, dtype=np.float32)

        if request.task == "classification":
            return self._run_classification(request, X_ctx, y_ctx, X_query)
        else:
            return self._run_regression(request, X_ctx, y_ctx, X_query)

    def _run_classification(
        self,
        request: TabularRequest,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> TabularResponse:
        clf = self._classifier_cls(
            device=self._device,
            n_estimators=self._n_estimators,
            inference_precision=self._inference_precision,
            categorical_features_indices=request.categorical_feature_indices,
        )
        clf.fit(X_ctx, y_ctx)
        preds = clf.predict(X_query)

        probabilities = None
        if request.output_mode == "probabilities":
            probabilities = clf.predict_proba(X_query).tolist()

        return TabularResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            task=request.task,
            predictions=preds.tolist(),
            probabilities=probabilities,
            classes=clf.classes_.tolist(),
            n_context=len(request.context_X),
            n_query=len(request.query_X),
        )

    def _run_regression(
        self,
        request: TabularRequest,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> TabularResponse:
        reg = self._regressor_cls(
            device=self._device,
            n_estimators=self._n_estimators,
            inference_precision=self._inference_precision,
            categorical_features_indices=request.categorical_feature_indices,
        )
        reg.fit(X_ctx, y_ctx)
        mean_preds = reg.predict(X_query, output_type="mean")

        quantiles = None
        if request.output_mode == "quantiles":
            q_preds = np.array(
                reg.predict(
                    X_query,
                    output_type="quantiles",
                    quantiles=request.quantile_levels,
                )
            )
            # q_preds shape: [n_quantiles, n_query]
            quantiles = {
                str(q): q_preds[i].tolist()
                for i, q in enumerate(request.quantile_levels)
            }

        return TabularResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            task=request.task,
            predictions=mean_preds.tolist(),
            quantiles=quantiles,
            n_context=len(request.context_X),
            n_query=len(request.query_X),
        )
