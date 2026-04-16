"""Tests for sheaf.tracing — OTel spans, no-op shims, and configure_tracing."""

from __future__ import annotations

import sys
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hide_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make opentelemetry look like it's not installed."""
    for key in list(sys.modules):
        if key.startswith("opentelemetry"):
            monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.setitem(sys.modules, "opentelemetry", None)  # type: ignore[call-overload]


def _reset_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level _DISABLED so env-var patches take effect."""
    import sheaf.tracing as t

    monkeypatch.setattr(t, "_DISABLED", False)


# ---------------------------------------------------------------------------
# _NoopSpan
# ---------------------------------------------------------------------------


class TestNoopSpan:
    def test_set_attribute_no_crash(self) -> None:
        from sheaf.tracing import _NoopSpan

        span = _NoopSpan()
        span.set_attribute("key", "value")
        span.set_attribute("num", 42)

    def test_record_exception_no_crash(self) -> None:
        from sheaf.tracing import _NoopSpan

        span = _NoopSpan()
        span.record_exception(ValueError("boom"))

    def test_set_status_no_crash(self) -> None:
        from sheaf.tracing import _NoopSpan

        span = _NoopSpan()
        span.set_status("ERROR", "desc")

    def test_context_manager(self) -> None:
        from sheaf.tracing import _NoopSpan

        with _NoopSpan() as span:
            assert isinstance(span, _NoopSpan)


# ---------------------------------------------------------------------------
# _NoopTracer
# ---------------------------------------------------------------------------


class TestNoopTracer:
    def test_start_as_current_span_yields_noop_span(self) -> None:
        from sheaf.tracing import _NoopSpan, _NoopTracer

        tracer = _NoopTracer()
        with tracer.start_as_current_span("test.span") as span:
            assert isinstance(span, _NoopSpan)

    def test_span_no_crash_on_exception(self) -> None:
        from sheaf.tracing import _NoopTracer

        tracer = _NoopTracer()
        with pytest.raises(RuntimeError):
            with tracer.start_as_current_span("test.span"):
                raise RuntimeError("expected")


# ---------------------------------------------------------------------------
# get_tracer — no-op paths
# ---------------------------------------------------------------------------


class TestGetTracerAbsent:
    def test_returns_noop_when_otel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hide_otel(monkeypatch)
        _reset_tracing(monkeypatch)
        import sheaf.tracing as t

        tracer = t.get_tracer()
        assert isinstance(tracer, t._NoopTracer)

    def test_returns_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", True)
        tracer = t.get_tracer()
        assert isinstance(tracer, t._NoopTracer)


# ---------------------------------------------------------------------------
# record_exception — safe for NoopSpan
# ---------------------------------------------------------------------------


class TestRecordException:
    def test_noop_span_no_crash(self) -> None:
        from sheaf.tracing import _NoopSpan, record_exception

        record_exception(_NoopSpan(), ValueError("boom"))

    def test_noop_span_no_crash_when_otel_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _hide_otel(monkeypatch)
        from sheaf.tracing import _NoopSpan, record_exception

        record_exception(_NoopSpan(), RuntimeError("err"))


# ---------------------------------------------------------------------------
# trace_predict — context manager
# ---------------------------------------------------------------------------


class TestTracePredict:
    def test_yields_span(self) -> None:
        from sheaf.tracing import trace_predict

        with trace_predict("dep", "time_series", "my-model", "req-123") as span:
            assert span is not None

    def test_no_crash_when_otel_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide_otel(monkeypatch)
        _reset_tracing(monkeypatch)
        from sheaf.tracing import trace_predict

        with trace_predict("dep", "time_series", "my-model", "req-123"):
            pass

    def test_exception_propagates(self) -> None:
        from sheaf.tracing import trace_predict

        with pytest.raises(ValueError):
            with trace_predict("dep", "time_series", "my-model", "req-123"):
                raise ValueError("propagated")


# ---------------------------------------------------------------------------
# trace_span — sub-span helper
# ---------------------------------------------------------------------------


class TestTraceSpan:
    def test_yields_span(self) -> None:
        from sheaf.tracing import trace_span

        with trace_span("sheaf.feast.resolve", deployment="dep") as span:
            assert span is not None

    def test_exception_propagates_and_records(self) -> None:
        from sheaf.tracing import trace_span

        with pytest.raises(RuntimeError):
            with trace_span("sheaf.backend.infer"):
                raise RuntimeError("backend error")


# ---------------------------------------------------------------------------
# configure_tracing — no-op paths
# ---------------------------------------------------------------------------


class TestConfigureTracingNoOp:
    def test_no_crash_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("SHEAF_OTEL_CONSOLE", raising=False)
        t.configure_tracing()  # should be a no-op

    def test_no_crash_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", True)
        t.configure_tracing()

    def test_no_crash_otel_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _hide_otel(monkeypatch)
        monkeypatch.setenv("SHEAF_OTEL_CONSOLE", "1")
        import sheaf.tracing as t

        t.configure_tracing()


# ---------------------------------------------------------------------------
# Functional tests — only when opentelemetry-sdk is installed
# ---------------------------------------------------------------------------


def _has_otel_sdk() -> bool:
    try:
        import opentelemetry.sdk.trace  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_otel = pytest.mark.skipif(
    not _has_otel_sdk(), reason="opentelemetry-sdk not installed"
)


def _make_test_tracer() -> tuple[Any, Any]:
    """Return (tracer, exporter) using a local TracerProvider (no global state)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("sheaf"), exporter


@_skip_no_otel
class TestTracingFunctional:
    """Tests that run only when opentelemetry-sdk is available.

    Span tests inject a local TracerProvider via monkeypatch to avoid fighting
    OTel's global singleton (set_tracer_provider is a one-time operation per
    process, so tests cannot safely reset it between runs).
    """

    def test_configure_tracing_calls_set_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure_tracing() calls trace.set_tracer_provider with an SDK provider."""
        from unittest.mock import MagicMock, patch

        from opentelemetry.sdk.trace import TracerProvider

        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        monkeypatch.setenv("SHEAF_OTEL_CONSOLE", "1")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        fake_current = MagicMock()  # not an SdkTracerProvider

        with (
            patch("opentelemetry.trace.get_tracer_provider", return_value=fake_current),
            patch("opentelemetry.trace.set_tracer_provider") as mock_set,
        ):
            t.configure_tracing(service_name="test-sheaf")

        mock_set.assert_called_once()
        provider_arg = mock_set.call_args[0][0]
        assert isinstance(provider_arg, TracerProvider)

    def test_configure_tracing_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """configure_tracing() skips when an SDK TracerProvider is already set."""
        from unittest.mock import patch

        from opentelemetry.sdk.trace import TracerProvider

        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        monkeypatch.setenv("SHEAF_OTEL_CONSOLE", "1")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        existing = TracerProvider()
        with (
            patch("opentelemetry.trace.get_tracer_provider", return_value=existing),
            patch("opentelemetry.trace.set_tracer_provider") as mock_set,
        ):
            t.configure_tracing()  # existing SDK provider → skip
        mock_set.assert_not_called()

    def test_get_tracer_returns_real_tracer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_tracer() returns a non-NoopTracer when OTel is installed."""
        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        # OTel is installed → get_tracer() always returns an OTel tracer
        tracer = t.get_tracer()
        assert not isinstance(tracer, t._NoopTracer)

    def test_trace_predict_creates_span_with_attributes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        tracer, exporter = _make_test_tracer()
        monkeypatch.setattr(t, "get_tracer", lambda: tracer)

        with t.trace_predict("mymodel", "time_series", "chronos", "abc-123"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "sheaf.predict"
        attrs = dict(span.attributes or {})
        assert attrs["sheaf.deployment"] == "mymodel"
        assert attrs["sheaf.model_type"] == "time_series"
        assert attrs["sheaf.model_name"] == "chronos"
        assert attrs["sheaf.request_id"] == "abc-123"

    def test_trace_span_sub_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        tracer, exporter = _make_test_tracer()
        monkeypatch.setattr(t, "get_tracer", lambda: tracer)

        with t.trace_span("sheaf.feast.resolve", deployment="dep"):
            pass

        spans = exporter.get_finished_spans()
        assert any(s.name == "sheaf.feast.resolve" for s in spans)

    def test_record_exception_sets_error_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.trace import StatusCode

        import sheaf.tracing as t

        monkeypatch.setattr(t, "_DISABLED", False)
        tracer, exporter = _make_test_tracer()

        with tracer.start_as_current_span("test") as span:
            t.record_exception(span, ValueError("fail"))

        finished = exporter.get_finished_spans()
        assert finished[0].status.status_code == StatusCode.ERROR
