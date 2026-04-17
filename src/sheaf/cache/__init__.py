"""In-process LRU response cache for Sheaf deployments.

Each deployment can opt in to caching via ``ModelSpec.cache``.  When enabled,
a SHA-256 of the canonical request JSON (excluding ``request_id`` and any
caller-specified fields) is used as the cache key.  Cache hits bypass inference
entirely — no batching, no backend call.

Usage::

    from sheaf.spec import ModelSpec, CacheConfig

    spec = ModelSpec(
        name="chronos-small",
        ...
        cache=CacheConfig(enabled=True, max_size=512, ttl_s=300.0),
    )

Good candidates for caching:
  - Embedding models (same image/text → same vector)
  - Time-series forecasts with fixed history
  - Tabular models with fixed feature rows

Poor candidates (non-deterministic or privacy-sensitive):
  - Diffusion with random seeds (include seed in the key — same seed IS same image)
  - Any model where the caller explicitly needs fresh output each time

SHEAF_CACHE_DISABLED=1 disables all caches regardless of config (useful in
smoke tests and integration runs where you want to exercise the backend).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, Field

_DISABLED = bool(os.environ.get("SHEAF_CACHE_DISABLED"))


class CacheConfig(BaseModel):
    """Cache configuration for a single deployment.

    Args:
        enabled:        Enable the cache (default ``False`` — opt-in).
        max_size:       Maximum number of LRU entries (default 1024).
        ttl_s:          Time-to-live in seconds.  ``None`` (default) means
                        entries never expire.  Set e.g. ``ttl_s=300`` to
                        expire cached forecasts after five minutes.
        exclude_fields: Request field names to omit from the cache key.
                        ``request_id`` is always excluded automatically.
                        Use this to exclude e.g. ``"seed"`` when you want
                        different diffusion seeds to produce distinct images
                        yet still benefit from caching same-seed repeats.
    """

    enabled: bool = False
    max_size: int = Field(default=1024, gt=0)
    ttl_s: float | None = Field(default=None, gt=0.0)
    exclude_fields: list[str] = Field(default_factory=list)


class ResponseCache:
    """Thread-safe in-process LRU cache for predict responses.

    Keys are SHA-256 hex digests of the canonical request JSON.
    Values are the serialised response dicts returned by ``model_dump``.

    Args:
        config: ``CacheConfig`` instance from the deployment's ``ModelSpec``.
    """

    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        # OrderedDict used as an LRU: oldest entry is first (left), newest last.
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    def make_key(self, deployment: str, request: Any) -> str:
        """Return a SHA-256 cache key for (deployment, request).

        ``request_id`` is always excluded (it is unique per call and must not
        affect the key).  Fields listed in ``config.exclude_fields`` are also
        dropped before hashing.

        Args:
            deployment: Deployment name (``ModelSpec.name``).
            request:    Any Pydantic ``BaseRequest`` subclass.

        Returns:
            64-character lowercase hex digest.
        """
        exclude = {"request_id"} | set(self._config.exclude_fields)
        d = request.model_dump(mode="json", exclude=exclude)
        # Prefix with deployment name so specs with the same input type but
        # different model weights don't share cache entries.
        raw = json.dumps({"_d": deployment, **d}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Get / set
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the cached response dict, or ``None`` on miss / expiry.

        Args:
            key: Cache key from :meth:`make_key`.
        """
        with self._lock:
            if key not in self._store:
                return None
            value, expiry = self._store[key]
            if expiry is not None and time.monotonic() > expiry:
                del self._store[key]
                return None
            # LRU bookkeeping: promote to most-recently-used position.
            self._store.move_to_end(key)
            return value  # type: ignore[return-value]

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a response dict under *key*.

        If the cache is at capacity the least-recently-used entry is evicted.

        Args:
            key:   Cache key from :meth:`make_key`.
            value: Serialised response dict (``response.model_dump(mode="json")``).
        """
        expiry = (
            time.monotonic() + self._config.ttl_s
            if self._config.ttl_s is not None
            else None
        )
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (value, expiry)
                return
            if len(self._store) >= self._config.max_size:
                self._store.popitem(last=False)  # evict LRU (first / oldest)
            self._store[key] = (value, expiry)

    # ------------------------------------------------------------------
    # Introspection (for tests / metrics)
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        with self._lock:
            return len(self._store)
