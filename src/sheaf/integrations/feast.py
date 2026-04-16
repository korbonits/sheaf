"""Feast feature store resolver for Sheaf serving.

Resolves a ``FeatureRef`` to a ``list[float]`` history before the request
reaches the model backend.  Designed to be initialised once at deployment
startup (``load()``) and then called concurrently per request (``resolve()``).

Usage (in a ``ModelSpec``)::

    spec = ModelSpec(
        name="chronos-production",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        feast_repo_path="/feast/feature_repo",
    )

Then send a request with ``feature_ref`` instead of raw ``history``::

    {
        "model_type": "time_series",
        "model_name": "chronos-production",
        "feature_ref": {
            "feature_view": "asset_prices",
            "feature_name": "close_history_30d",
            "entity_key": "ticker",
            "entity_value": "AAPL"
        },
        "horizon": 6,
        "frequency": "1d"
    }

The serving layer calls ``FeastResolver.resolve()`` to fetch the online
feature, sets ``request.history`` to the returned list, and passes the
resolved request to the backend — which never sees ``feature_ref``.

Notes:
    * The Feast feature must store the complete history as a ``list[float]``
      (or ``list[int]``, which is coerced).  Scalar features are not supported;
      use multiple rolling-window features assembled into a list.
    * Resolution is per-request (not per-batch) because each request may
      reference a different entity.
    * ``FeastResolver`` is thread-safe: ``FeatureStore.get_online_features``
      is stateless after initialisation.
    * Install with ``pip install 'sheaf-serve[feast]'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sheaf.api.time_series import FeatureRef

_logger = logging.getLogger(__name__)


class FeastResolver:
    """Resolves Feast online features to a flat ``list[float]`` history.

    Args:
        repo_path: Path to the Feast feature repo directory (the directory
            that contains ``feature_store.yaml``).
    """

    def __init__(self, repo_path: str) -> None:
        self._repo_path = repo_path
        self._store: Any = None

    def load(self) -> None:
        """Initialise the Feast ``FeatureStore``.

        Called once at deployment startup.  Feast reads ``feature_store.yaml``
        from ``repo_path`` and connects to the configured online store (Redis,
        DynamoDB, SQLite, etc.).

        Raises:
            ImportError: if ``feast`` is not installed.
        """
        try:
            from feast import FeatureStore  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError(
                "feast is required for FeastResolver. "
                "Install it with: pip install 'sheaf-serve[feast]'"
            ) from exc

        self._store = FeatureStore(repo_path=self._repo_path)
        _logger.info("FeastResolver initialised (repo: %s)", self._repo_path)

    def resolve(self, ref: FeatureRef) -> list[float]:
        """Fetch online features for a single entity and return as history.

        Args:
            ref: ``FeatureRef`` identifying the feature view, feature column,
                entity key, and entity value.

        Returns:
            Flat ``list[float]`` ready to use as ``TimeSeriesRequest.history``.

        Raises:
            RuntimeError: if ``load()`` has not been called.
            ValueError: if the feature is not found in the response, or if the
                feature value is not a list (scalar features are not supported).
        """
        if self._store is None:
            raise RuntimeError(
                "FeastResolver.load() must be called before resolve(). "
                "It is called automatically by the serving layer at startup."
            )

        result: dict[str, list[Any]] = self._store.get_online_features(
            features=[f"{ref.feature_view}:{ref.feature_name}"],
            entity_rows=[{ref.entity_key: ref.entity_value}],
        ).to_dict()

        values = result.get(ref.feature_name)
        if values is None:
            raise ValueError(
                f"Feature '{ref.feature_view}:{ref.feature_name}' not found in "
                f"Feast response. Available keys: {sorted(result)}"
            )

        # get_online_features returns one row per entity_row; we requested one.
        history = values[0]
        if not isinstance(history, list):
            raise ValueError(
                f"Feature '{ref.feature_name}' must return list[float] to be used "
                f"as time series history, got {type(history).__name__}: {history!r}. "
                "Store the full sequence as an array feature."
            )

        return [float(x) for x in history]
