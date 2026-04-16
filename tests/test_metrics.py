"""Tests for sheaf.metrics — Prometheus counters, histograms, and /metrics endpoint."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_metrics() -> Any:
    """Force-reload sheaf.metrics so module-level _DISABLED is re-evaluated."""
    mod = sys.modules.get("sheaf.metrics")
    if mod is not None:
        importlib.reload(mod)
    return importlib.import_module("sheaf.metrics")


# ---------------------------------------------------------------------------
# No-ops when prometheus_client is absent
# ---------------------------------------------------------------------------


class TestMetricsAbsent:
    """All public functions must be no-ops when prometheus_client is not installed."""

    def setup_method(self) -> None:
        # Hide prometheus_client from imports for the duration of the test.
        self._orig = sys.modules.get("prometheus_client", None)
        sys.modules["prometheus_client"] = None  # type: ignore[assignment]
        # Clear cached registry so _get_registry() re-runs the import.
        import sheaf.metrics as m

        m._registry = None
        m._counters_cache.clear()

    def teardown_method(self) -> None:
        if self._orig is None:
            sys.modules.pop("prometheus_client", None)
        else:
            sys.modules["prometheus_client"] = self._orig
        import sheaf.metrics as m

        m._registry = None
        m._counters_cache.clear()

    def test_record_predict_no_crash(self) -> None:
        from sheaf.metrics import record_predict

        record_predict("dep", "time_series", "ok", 0.1)  # must not raise

    def test_record_batch_no_crash(self) -> None:
        from sheaf.metrics import record_batch

        record_batch("dep", 4)

    def test_record_load_no_crash(self) -> None:
        from sheaf.metrics import record_load

        record_load("dep", "time_series", 1.5)

    def test_time_load_context_no_crash(self) -> None:
        from sheaf.metrics import time_load

        with time_load("dep", "time_series"):
            pass

    def test_register_metrics_endpoint_no_crash(self) -> None:
        from fastapi import FastAPI

        from sheaf.metrics import register_metrics_endpoint

        app = FastAPI()
        register_metrics_endpoint(app, "dep")  # must be a no-op
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/metrics" not in routes


# ---------------------------------------------------------------------------
# No-ops when SHEAF_METRICS_DISABLED=1
# ---------------------------------------------------------------------------


class TestMetricsDisabled:
    def setup_method(self) -> None:
        os.environ["SHEAF_METRICS_DISABLED"] = "1"

    def teardown_method(self) -> None:
        os.environ.pop("SHEAF_METRICS_DISABLED", None)
        # Re-set _DISABLED to False so subsequent tests are unaffected.
        import sheaf.metrics as m

        m._DISABLED = False
        m._registry = None
        m._counters_cache.clear()

    def test_record_predict_no_crash(self) -> None:
        import sheaf.metrics as m

        m._DISABLED = True
        m.record_predict("dep", "time_series", "ok", 0.1)

    def test_register_metrics_no_route(self) -> None:
        from fastapi import FastAPI

        import sheaf.metrics as m

        m._DISABLED = True
        app = FastAPI()
        m.register_metrics_endpoint(app, "dep")
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/metrics" not in routes


# ---------------------------------------------------------------------------
# Functional tests with real prometheus_client (if installed)
# ---------------------------------------------------------------------------


def _has_prometheus() -> bool:
    try:
        import prometheus_client  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_prometheus = pytest.mark.skipif(
    not _has_prometheus(), reason="prometheus_client not installed"
)


@_skip_no_prometheus
class TestMetricsFunctional:
    """Tests that run only when prometheus_client is actually available."""

    def setup_method(self) -> None:
        import sheaf.metrics as m

        m._DISABLED = False
        m._registry = None
        m._counters_cache.clear()

    def teardown_method(self) -> None:
        import sheaf.metrics as m

        m._registry = None
        m._counters_cache.clear()

    def test_record_predict_ok_increments_counter(self) -> None:
        import sheaf.metrics as m

        m.record_predict("mymodel", "time_series", "ok", 0.05)
        counters = m._counters()
        assert counters is not None
        val = (
            counters["requests_total"]
            .labels(deployment="mymodel", model_type="time_series", status="ok")
            ._value.get()
        )
        assert val == 1.0

    def test_record_predict_error_increments_counter(self) -> None:
        import sheaf.metrics as m

        m.record_predict("mymodel", "time_series", "error", 0.2)
        counters = m._counters()
        assert counters is not None
        val = (
            counters["requests_total"]
            .labels(deployment="mymodel", model_type="time_series", status="error")
            ._value.get()
        )
        assert val == 1.0

    def test_record_predict_observes_histogram(self) -> None:
        import sheaf.metrics as m

        m.record_predict("mymodel", "time_series", "ok", 0.5)
        counters = m._counters()
        assert counters is not None
        hist = counters["request_duration"].labels(
            deployment="mymodel", model_type="time_series"
        )
        assert hist._sum.get() == pytest.approx(0.5)

    def test_record_batch_observes_histogram(self) -> None:
        import sheaf.metrics as m

        m.record_batch("mymodel", 8)
        counters = m._counters()
        assert counters is not None
        hist = counters["batch_size"].labels(deployment="mymodel")
        assert hist._sum.get() == pytest.approx(8.0)

    def test_record_load_sets_gauge(self) -> None:
        import sheaf.metrics as m

        m.record_load("mymodel", "time_series", 3.7)
        counters = m._counters()
        assert counters is not None
        gauge = counters["backend_load_seconds"].labels(
            deployment="mymodel", model_type="time_series"
        )
        assert gauge._value.get() == pytest.approx(3.7)

    def test_time_load_records_elapsed(self) -> None:
        import sheaf.metrics as m

        with m.time_load("mymodel", "time_series"):
            pass  # effectively 0 seconds
        counters = m._counters()
        assert counters is not None
        gauge = counters["backend_load_seconds"].labels(
            deployment="mymodel", model_type="time_series"
        )
        assert gauge._value.get() >= 0.0

    def test_register_metrics_endpoint_adds_route(self) -> None:
        from fastapi import FastAPI

        import sheaf.metrics as m

        app = FastAPI()
        m.register_metrics_endpoint(app, "mymodel")
        routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
        assert "/metrics" in routes

    def test_metrics_endpoint_returns_200(self) -> None:
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import sheaf.metrics as m

        app = FastAPI()
        m.register_metrics_endpoint(app, "mymodel")
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_endpoint_content_type(self) -> None:
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        import sheaf.metrics as m

        app = FastAPI()
        m.register_metrics_endpoint(app, "mymodel")
        client = TestClient(app)
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_counters_idempotent_second_call(self) -> None:
        """Calling _counters() twice returns the same dict (no re-registration)."""
        import sheaf.metrics as m

        c1 = m._counters()
        c2 = m._counters()
        assert c1 is c2
