"""Prometheus metrics for Sheaf deployments.

Exposes a ``/metrics`` endpoint on each deployment that a Prometheus scraper
can poll.  All metrics are labelled by ``deployment`` and ``model_type`` so a
single Prometheus instance can fan-in metrics from every model.

Metrics exported
----------------
sheaf_requests_total          counter   requests by deployment/model_type/status
sheaf_request_duration_seconds histogram request latency in seconds
sheaf_batch_size_total        histogram requests grouped per @serve.batch call
sheaf_backend_load_seconds    gauge     model load time at startup (seconds)

Usage::

    # Enabled automatically when prometheus-client is installed.
    # Install with: pip install 'sheaf-serve[metrics]'
    #
    # The /metrics endpoint is registered on each deployment's FastAPI app by
    # calling register_metrics_endpoint(app) at startup.  record_predict() and
    # record_batch() are called from the inference hot-path.

SHEAF_METRICS_DISABLED=1 disables the endpoint and all recording even when
prometheus-client is installed (useful in test environments).
"""

from __future__ import annotations

import os
import time
from typing import Any

_DISABLED = bool(os.environ.get("SHEAF_METRICS_DISABLED"))

# ---------------------------------------------------------------------------
# Lazy registry — only imported when prometheus_client is available
# ---------------------------------------------------------------------------

_registry: Any = None  # prometheus_client.CollectorRegistry or None


def _get_registry() -> Any:
    global _registry
    if _registry is not None:
        return _registry
    try:
        from prometheus_client import CollectorRegistry

        _registry = CollectorRegistry()
    except ImportError:
        _registry = None
    return _registry


# Metric singletons — populated on first call to _counters()
_counters_cache: dict[str, Any] = {}


def _counters() -> dict[str, Any] | None:
    """Return the metric objects, creating them once on first access.

    Returns None if prometheus_client is not installed or metrics are disabled.
    """
    if _DISABLED:
        return None
    reg = _get_registry()
    if reg is None:
        return None
    if _counters_cache:
        return _counters_cache

    from prometheus_client import Counter, Gauge, Histogram

    _counters_cache["requests_total"] = Counter(
        "sheaf_requests_total",
        "Total predict requests",
        ["deployment", "model_type", "status"],
        registry=reg,
    )
    _counters_cache["request_duration"] = Histogram(
        "sheaf_request_duration_seconds",
        "Predict request latency in seconds",
        ["deployment", "model_type"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=reg,
    )
    _counters_cache["batch_size"] = Histogram(
        "sheaf_batch_size_total",
        "Number of requests grouped per batch call",
        ["deployment"],
        buckets=(1, 2, 4, 8, 16, 32, 64),
        registry=reg,
    )
    _counters_cache["backend_load_seconds"] = Gauge(
        "sheaf_backend_load_seconds",
        "Backend model load time in seconds",
        ["deployment", "model_type"],
        registry=reg,
    )
    return _counters_cache


# ---------------------------------------------------------------------------
# Public API — called from server.py / modal_server.py
# ---------------------------------------------------------------------------


def record_predict(
    deployment: str,
    model_type: str,
    status: str,
    latency_s: float,
) -> None:
    """Record a completed predict call (ok or error).

    Args:
        deployment: Deployment / model name (``ModelSpec.name``).
        model_type: ``ModelType`` string (``"time_series"``, ``"embedding"``, …).
        status:     ``"ok"`` or ``"error"``.
        latency_s:  Elapsed time in seconds (use ``time.perf_counter()`` delta).
    """
    m = _counters()
    if m is None:
        return
    m["requests_total"].labels(
        deployment=deployment, model_type=model_type, status=status
    ).inc()
    m["request_duration"].labels(deployment=deployment, model_type=model_type).observe(
        latency_s
    )


def record_batch(deployment: str, batch_size: int) -> None:
    """Record the size of a batch delivered to ``@serve.batch`` / ``async_predict``.

    Args:
        deployment: Deployment name.
        batch_size: Number of requests in this batch.
    """
    m = _counters()
    if m is None:
        return
    m["batch_size"].labels(deployment=deployment).observe(batch_size)


def record_load(deployment: str, model_type: str, load_time_s: float) -> None:
    """Record backend model load time.

    Args:
        deployment:  Deployment name.
        model_type:  ModelType string.
        load_time_s: Elapsed time for ``backend.load()`` in seconds.
    """
    m = _counters()
    if m is None:
        return
    m["backend_load_seconds"].labels(deployment=deployment, model_type=model_type).set(
        load_time_s
    )


def time_load(deployment: str, model_type: str) -> Any:
    """Context manager that times a block and calls ``record_load`` on exit.

    Usage::

        with time_load(spec.name, spec.model_type):
            self._backend.load()
    """
    return _LoadTimer(deployment, model_type)


class _LoadTimer:
    def __init__(self, deployment: str, model_type: str) -> None:
        self._deployment = deployment
        self._model_type = model_type
        self._t0: float = 0.0

    def __enter__(self) -> _LoadTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed = time.perf_counter() - self._t0
        record_load(self._deployment, self._model_type, elapsed)


def register_metrics_endpoint(app: Any, deployment: str) -> None:
    """Add ``GET /metrics`` to a FastAPI app.

    A no-op if prometheus_client is not installed or
    ``SHEAF_METRICS_DISABLED=1`` is set.

    Args:
        app:        The ``fastapi.FastAPI`` instance.
        deployment: Deployment name — included in the ``target`` label when
                    the endpoint is scraped by Prometheus.
    """
    if _DISABLED or _get_registry() is None:
        return

    from starlette.responses import Response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        # Import and fetch lazily so this closure captures no non-picklable
        # objects (e.g. threading.Lock inside CollectorRegistry).  Ray Serve
        # serialises the FastAPI app at class-definition time via cloudpickle,
        # so any closure over prometheus_client internals would break pickling.
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(
            generate_latest(_get_registry()), media_type=CONTENT_TYPE_LATEST
        )
