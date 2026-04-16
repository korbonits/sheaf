"""Tests for sheaf.logging — JsonFormatter and configure_logging."""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import patch

import pytest

from sheaf.api.base import ModelType
from sheaf.logging import JsonFormatter, configure_logging
from sheaf.spec import ModelSpec

# ---------------------------------------------------------------------------
# JsonFormatter unit tests
# ---------------------------------------------------------------------------


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    extra: dict | None = None,
    exc_info: bool = False,
) -> logging.LogRecord:
    logger = logging.getLogger("test.logger")
    record = logger.makeRecord(
        name="test.logger",
        level=level,
        fn="test_logging.py",
        lno=0,
        msg=msg,
        args=(),
        exc_info=(ValueError, ValueError("boom"), None) if exc_info else None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


class TestJsonFormatter:
    def test_output_is_valid_json(self) -> None:
        fmt = JsonFormatter()
        output = fmt.format(_make_record("hi"))
        doc = json.loads(output)
        assert isinstance(doc, dict)

    def test_standard_fields_present(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record("hello world")))
        assert doc["level"] == "INFO"
        assert doc["logger"] == "test.logger"
        assert doc["message"] == "hello world"
        assert "ts" in doc

    def test_ts_format(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record()))
        # e.g. "2026-04-16T23:30:00.123Z"
        ts = doc["ts"]
        assert "T" in ts
        assert ts.endswith("Z")
        assert "." in ts

    def test_extra_fields_included(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(
            fmt.format(
                _make_record(
                    extra={
                        "request_id": "abc-123",
                        "deployment": "chronos",
                        "latency_ms": 42.1,
                        "status": "ok",
                    }
                )
            )
        )
        assert doc["request_id"] == "abc-123"
        assert doc["deployment"] == "chronos"
        assert doc["latency_ms"] == 42.1
        assert doc["status"] == "ok"

    def test_exception_info_included(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record(exc_info=True)))
        assert "exc" in doc
        assert "ValueError" in doc["exc"]
        assert "boom" in doc["exc"]

    def test_no_exception_no_exc_key(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record()))
        assert "exc" not in doc

    def test_warning_level(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record(level=logging.WARNING)))
        assert doc["level"] == "WARNING"

    def test_error_level(self) -> None:
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record(level=logging.ERROR)))
        assert doc["level"] == "ERROR"

    def test_internal_record_attrs_not_leaked(self) -> None:
        """lineno, pathname, etc. should not appear in the JSON output."""
        fmt = JsonFormatter()
        doc = json.loads(fmt.format(_make_record()))
        for internal in ("lineno", "pathname", "filename", "funcName", "thread"):
            assert internal not in doc, f"internal field {internal!r} leaked into JSON"

    def test_non_serialisable_value_uses_str_fallback(self) -> None:
        """default=str means non-JSON-serialisable extras don't crash."""
        fmt = JsonFormatter()

        class _Unserializable:
            pass

        record = _make_record(extra={"obj": _Unserializable()})
        output = fmt.format(record)
        doc = json.loads(output)
        assert "obj" in doc  # serialised as str(...)


# ---------------------------------------------------------------------------
# configure_logging tests
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def _json_handlers(self) -> list[logging.Handler]:
        root = logging.getLogger()
        return [h for h in root.handlers if isinstance(h.formatter, JsonFormatter)]

    def setup_method(self) -> None:
        root = logging.getLogger()
        root.handlers = [
            h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)
        ]

    def teardown_method(self) -> None:
        root = logging.getLogger()
        root.handlers = [
            h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)
        ]

    def test_adds_json_handler(self) -> None:
        configure_logging()
        assert len(self._json_handlers()) == 1

    def test_idempotent_no_duplicate_handlers(self) -> None:
        configure_logging()
        configure_logging()
        configure_logging()
        assert len(self._json_handlers()) == 1

    def test_sets_root_level(self) -> None:
        configure_logging(level=logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG


# ---------------------------------------------------------------------------
# Server integration — request_id and latency flow through caplog
# (tested via _build_asgi_app, which is a plain FastAPI app, not Ray-wrapped)
# ---------------------------------------------------------------------------


class TestServerStructuredLogging:
    def _make_spec(self, name: str = "test-deploy") -> object:
        from sheaf.api.base import ModelType
        from sheaf.spec import ModelSpec

        return ModelSpec(
            name=name,
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
        )

    def test_predict_ok_logs_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        import json

        from starlette.testclient import TestClient

        from sheaf.modal_server import _build_asgi_app

        with (
            patch.dict(os.environ, {"SHEAF_EXTRA_BACKENDS": "tests.stubs"}),
            caplog.at_level(logging.INFO, logger="sheaf.modal_server"),
        ):
            app = _build_asgi_app([self._make_spec()])
            client = TestClient(app)
            resp = client.post(
                "/test-deploy/predict",
                content=json.dumps(
                    {
                        "model_type": "time_series",
                        "model_name": "test",
                        "history": [1.0, 2.0],
                        "horizon": 1,
                        "frequency": "1d",
                    }
                ),
                headers={"Content-Type": "application/json"},
            )

        assert resp.status_code == 200
        response_body = resp.json()
        request_id_str = response_body["request_id"]

        rid = request_id_str
        matching = [r for r in caplog.records if getattr(r, "request_id", None) == rid]
        assert matching, "No log record with matching request_id found"
        record = matching[0]
        assert getattr(record, "status") == "ok"
        assert getattr(record, "deployment") == "test-deploy"
        assert isinstance(getattr(record, "latency_ms"), float)

    def test_predict_error_logs_status_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import json

        from starlette.testclient import TestClient

        from sheaf.modal_server import _build_asgi_app

        with (
            patch.dict(os.environ, {"SHEAF_EXTRA_BACKENDS": "tests.stubs"}),
            caplog.at_level(logging.ERROR, logger="sheaf.modal_server"),
        ):
            app = _build_asgi_app(
                [
                    ModelSpec(
                        name="error-deploy",
                        model_type=ModelType.TIME_SERIES,
                        backend="_smoke_error",
                    )
                ]
            )
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/error-deploy/predict",
                content=json.dumps(
                    {
                        "model_type": "time_series",
                        "model_name": "test",
                        "history": [1.0],
                        "horizon": 1,
                        "frequency": "1d",
                    }
                ),
                headers={"Content-Type": "application/json"},
            )

        assert resp.status_code == 500
        error_records = [
            r for r in caplog.records if getattr(r, "status", None) == "error"
        ]
        assert error_records, "No error-status log record found"
        record = error_records[0]
        assert getattr(record, "deployment") == "error-deploy"
        assert "RuntimeError" in getattr(record, "error", "")
        assert isinstance(getattr(record, "latency_ms"), float)
