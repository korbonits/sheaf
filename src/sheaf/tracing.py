"""OpenTelemetry distributed tracing for Sheaf deployments.

Each predict request is wrapped in a ``sheaf.predict`` span with attributes
for deployment, model_type, model_name, and request_id.  Sub-spans are emitted
for Feast feature resolution (``sheaf.feast.resolve``) and backend inference
(``sheaf.backend.infer``).

Usage::

    # Enabled when opentelemetry-sdk is installed and an exporter is configured.
    # Install with: pip install 'sheaf-serve[tracing]'
    #
    # Configure the exporter via standard OTel env vars, then call
    # configure_tracing() once at startup (ModelServer.run() / _build_asgi_app):
    #
    #   OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
    #   OTEL_SERVICE_NAME=my-sheaf
    #
    # For local console output (dev / CI):
    #   SHEAF_OTEL_CONSOLE=1

SHEAF_TRACING_DISABLED=1 disables all tracing even when opentelemetry is installed.

If you configure your own TracerProvider before calling configure_tracing(), the
existing provider is left in place and configure_tracing() is a no-op.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_DISABLED = bool(os.environ.get("SHEAF_TRACING_DISABLED"))
_TRACER_NAME = "sheaf"


# ---------------------------------------------------------------------------
# No-op shims — used when opentelemetry is absent or tracing is disabled
# ---------------------------------------------------------------------------


class _NoopSpan:
    """Minimal span shim — all methods are silent no-ops."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: BaseException, **kwargs: Any) -> None:
        pass

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(
        self, name: str, **kwargs: Any
    ) -> Generator[_NoopSpan, None, None]:
        yield _NoopSpan()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tracer() -> Any:
    """Return an OTel Tracer, or a ``_NoopTracer`` shim if unavailable/disabled.

    Always safe to call — the returned object has the same interface whether or
    not ``opentelemetry`` is installed.
    """
    if _DISABLED:
        return _NoopTracer()
    try:
        from opentelemetry import trace  # ty: ignore[unresolved-import]

        return trace.get_tracer(_TRACER_NAME)
    except ImportError:
        return _NoopTracer()


def configure_tracing(service_name: str = "sheaf") -> None:
    """Set up the OTel SDK from environment variables.

    Call once at startup (``ModelServer.run()`` / ``_build_asgi_app``).

    A no-op when any of the following is true:

    - ``opentelemetry-sdk`` is not installed
    - ``SHEAF_TRACING_DISABLED=1`` is set
    - Neither ``OTEL_EXPORTER_OTLP_ENDPOINT`` nor ``SHEAF_OTEL_CONSOLE=1`` is set
    - An SDK ``TracerProvider`` is already configured (idempotent)

    Exporters configured:

    - **OTLP/HTTP** when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set
      (e.g. ``http://jaeger:4318``)
    - **Console** when ``SHEAF_OTEL_CONSOLE=1`` (human-readable, dev/CI)

    Args:
        service_name: Value for the ``service.name`` OTel resource attribute.
            Defaults to ``"sheaf"``; override with ``OTEL_SERVICE_NAME`` env var
            (standard OTel) or pass explicitly.
    """
    if _DISABLED:
        return

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    console = os.environ.get("SHEAF_OTEL_CONSOLE")
    if not otlp_endpoint and not console:
        return  # Nothing to export to — skip SDK initialisation entirely.

    try:
        from opentelemetry import trace  # ty: ignore[unresolved-import]
        from opentelemetry.sdk.resources import (  # ty: ignore[unresolved-import]
            SERVICE_NAME,
            Resource,
        )
        from opentelemetry.sdk.trace import (  # ty: ignore[unresolved-import]
            TracerProvider as SdkTracerProvider,
        )
    except ImportError:
        return  # opentelemetry-sdk not installed

    # Idempotent: honour an existing SDK provider (e.g. set by the caller).
    current = trace.get_tracer_provider()
    if isinstance(current, SdkTracerProvider):
        return

    resource = Resource.create(
        {SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", service_name)}
    )
    provider = SdkTracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # ty: ignore[unresolved-import]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import (  # ty: ignore[unresolved-import]
                BatchSpanProcessor,
            )

            exporter = OTLPSpanExporter(
                endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces"
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass  # OTLP HTTP exporter not installed — skip silently

    if console:
        try:
            from opentelemetry.sdk.trace.export import (  # ty: ignore[unresolved-import]
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        except ImportError:
            pass

    trace.set_tracer_provider(provider)


def record_exception(span: Any, exc: BaseException) -> None:
    """Record *exc* on *span* and set status to ERROR.

    Safe to call with a ``_NoopSpan`` — all methods are no-ops.
    Handles the ``StatusCode`` import guard so call sites don't need to.

    Args:
        span: An OTel ``Span`` or ``_NoopSpan``.
        exc:  The exception to record.
    """
    span.record_exception(exc)
    try:
        from opentelemetry.trace import StatusCode  # ty: ignore[unresolved-import]

        span.set_status(StatusCode.ERROR, str(exc))
    except ImportError:
        pass


@contextmanager
def trace_predict(
    deployment: str,
    model_type: str,
    model_name: str,
    request_id: str,
) -> Generator[Any, None, None]:
    """Context manager wrapping a predict call in a ``sheaf.predict`` span.

    Sets standard attributes and yields the span so callers can add custom
    attributes or access the span for sub-span nesting.  Does **not**
    automatically record exceptions — call :func:`record_exception` explicitly
    in the ``except`` clause so the original exception is captured before it
    is re-raised as an ``HTTPException``.

    Example::

        with trace_predict(name, request.model_type,
                           request.model_name or "", str(request.request_id)) as span:
            try:
                result = await backend.async_predict(request)
            except Exception as exc:
                record_exception(span, exc)
                raise HTTPException(status_code=500, ...) from exc
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("sheaf.predict") as span:
        span.set_attribute("sheaf.deployment", deployment)
        span.set_attribute("sheaf.model_type", str(model_type))
        span.set_attribute("sheaf.model_name", model_name or "")
        span.set_attribute("sheaf.request_id", request_id)
        yield span


@contextmanager
def trace_span(name: str, **attrs: Any) -> Generator[Any, None, None]:
    """Context manager for an arbitrary sub-span with keyword attributes.

    Used for Feast resolution (``sheaf.feast.resolve``) and backend inference
    (``sheaf.backend.infer``).  Automatically records exceptions and sets
    ERROR status if the body raises.

    Example::

        with trace_span("sheaf.feast.resolve", deployment=name):
            history = feast.resolve(feature_ref)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            record_exception(span, exc)
            raise
