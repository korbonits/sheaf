"""Tests for sheaf.openapi.generate() and the python -m sheaf.openapi CLI.

Verifies that the OpenAPI schema reflects deployments correctly without
loading real model weights, that request/response types resolve, and that
the CLI handles dotted-path resolution + IO.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

import tests.stubs  # noqa: F401 — registers smoke backends
from sheaf.api.base import ModelType
from sheaf.openapi import generate, main
from sheaf.spec import ModelSpec, ResourceConfig
from tests.stubs import SmokeEmbeddingBackend, SmokeTimeSeriesBackend


def _ts_spec(name: str = "ts") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.TIME_SERIES,
        backend="_smoke_ts",
        backend_cls=SmokeTimeSeriesBackend,
        resources=ResourceConfig(num_cpus=1),
    )


def _emb_spec(name: str = "emb") -> ModelSpec:
    return ModelSpec(
        name=name,
        model_type=ModelType.EMBEDDING,
        backend="_smoke_embedding",
        backend_cls=SmokeEmbeddingBackend,
        resources=ResourceConfig(num_cpus=1),
    )


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_openapi_dict(self) -> None:
        schema = generate([_ts_spec()])
        assert schema["openapi"].startswith("3.")
        assert "paths" in schema
        assert "components" in schema

    def test_routes_in_schema(self) -> None:
        """ModalServer uses templated `/{name}/...` paths — schema is the
        same regardless of which deployments are configured."""
        schema = generate([_ts_spec("alpha"), _emb_spec("beta")])
        paths = schema["paths"]
        assert "/{name}/predict" in paths
        assert "/{name}/stream" in paths
        assert "/{name}/health" in paths
        assert "/{name}/ready" in paths

    def test_request_and_response_schemas_resolve(self) -> None:
        """Pydantic request/response types must appear in components.schemas.

        Because ``predict`` is typed as ``AnyRequest`` and the response is
        typed via a discriminated union, every member type ends up referenced
        in ``components.schemas`` regardless of which deployments are passed.
        """
        schema = generate([_ts_spec()])
        components = schema["components"]["schemas"]
        # Every union member is referenced — pick a few to assert
        assert "TimeSeriesRequest" in components
        assert "DiffusionRequest" in components
        assert "EmbeddingRequest" in components

    def test_empty_specs_still_produces_schema(self) -> None:
        """The route shape is global; an empty deployment list is valid."""
        schema = generate([])
        assert "/{name}/predict" in schema["paths"]

    def test_does_not_call_backend_load(self) -> None:
        """generate() must not invoke backend.load() — proven via a backend
        that would raise if loaded."""

        class _ExplodingBackend:
            def load(self) -> None:
                raise RuntimeError("don't load me")

            @property
            def model_type(self) -> str:
                return ModelType.TIME_SERIES.value

            def predict(self, request):  # type: ignore[no-untyped-def]
                return None

        spec = ModelSpec(
            name="boom",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",  # registry key (won't be used for instantiation)
            backend_cls=_ExplodingBackend,  # cls overrides the registry lookup
            resources=ResourceConfig(num_cpus=1),
        )
        schema = generate([spec])  # must not raise
        assert "/{name}/predict" in schema["paths"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_specs_module(tmp_path: Path) -> Path:
    """Write a tiny module exposing ``specs: list[ModelSpec]`` for the CLI."""
    mod_path = tmp_path / "my_test_specs.py"
    mod_path.write_text(
        textwrap.dedent(
            """
            from sheaf.api.base import ModelType
            from sheaf.spec import ModelSpec, ResourceConfig
            import tests.stubs  # registers _smoke_ts
            from tests.stubs import SmokeTimeSeriesBackend

            specs = [
                ModelSpec(
                    name="cli-ts",
                    model_type=ModelType.TIME_SERIES,
                    backend="_smoke_ts",
                    backend_cls=SmokeTimeSeriesBackend,
                    resources=ResourceConfig(num_cpus=1),
                ),
            ]
            """
        )
    )
    return mod_path


class TestCLI:
    def test_writes_to_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_specs_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("my_test_specs", None)
        out = tmp_path / "out.json"

        rc = main(["--specs", "my_test_specs:specs", "--out", str(out)])

        assert rc == 0
        schema = json.loads(out.read_text())
        assert "/{name}/predict" in schema["paths"]

    def test_writes_to_stdout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        _write_specs_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("my_test_specs", None)

        rc = main(["--specs", "my_test_specs:specs"])

        assert rc == 0
        captured = capsys.readouterr()
        schema = json.loads(captured.out)
        assert "/{name}/predict" in schema["paths"]

    def test_invalid_specs_path_returns_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        rc = main(["--specs", "no.such.module:specs"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "error" in captured.err

    def test_missing_colon_returns_2(self, capsys: pytest.CaptureFixture) -> None:
        rc = main(["--specs", "no_colon"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "module.path:attribute" in captured.err
