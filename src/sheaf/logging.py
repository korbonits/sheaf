"""Structured JSON logging for Sheaf deployments.

Usage::

    # At server startup (ModelServer.run() / _build_asgi_app):
    from sheaf.logging import configure_logging
    configure_logging()

    # Then use standard logging — extra fields flow into JSON output:
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "predict ok",
        extra={
            "request_id": "abc-123",
            "deployment": "chronos",
            "latency_ms": 42.1,
        },
    )

Output (one line per record)::

    {"ts": "2026-04-16T23:30:00.123Z", "level": "INFO", "logger": "sheaf.server",
     "message": "predict ok", "request_id": "abc-123", "deployment": "chronos",
     "latency_ms": 42.1}

Enable automatically by setting ``SHEAF_LOG_JSON=1`` in the environment.
"""

from __future__ import annotations

import json
import logging

# Canonical set of LogRecord attributes that should NOT be copied into the
# JSON document as extra fields (they are either redundant with the top-level
# keys we already emit, or internal Python bookkeeping).
_RECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {
    "message",  # already emitted as "message"
    "asctime",  # replaced by "ts"
    "taskName",  # Python 3.12+ addition; redundant
}


class JsonFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object.

    Standard fields emitted:
        ts        — ISO-8601 timestamp with milliseconds (UTC marker appended)
        level     — levelname (INFO, WARNING, ERROR, …)
        logger    — logger name
        message   — formatted log message

    Any key passed via ``extra={}`` in the logging call is included verbatim.
    Exception info (if present) is serialised under the ``exc`` key.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        base = self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S")
        ts = f"{base}.{int(record.msecs):03d}Z"
        doc: dict[str, object] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }
        # Inject caller-supplied extra fields
        for key, val in record.__dict__.items():
            if key not in _RECORD_ATTRS and not key.startswith("_"):
                doc[key] = val
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Install JSON formatting on the root logger.

    Idempotent — calling multiple times does not add duplicate handlers.
    A no-op if a ``JsonFormatter`` handler is already attached.

    Args:
        level: Logging level for the root logger (default ``logging.INFO``).
    """
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
