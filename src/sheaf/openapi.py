"""Generate the OpenAPI 3 spec for a sheaf deployment.

Useful for:
  - Publishing a static schema teams can use to generate non-Python clients
    (TypeScript, Go, …) via ``openapi-python-client``, ``openapi-generator``,
    etc.
  - Bundling the spec into release artifacts so the contract is versioned.

The schema is dynamic — it depends on which deployments (``ModelSpec``s) are
configured, since each becomes its own ``/<name>/predict`` route.  Backends
are NOT loaded during generation (``load_backends=False``); we only need the
route shape and the Pydantic request/response types, both of which are
available without GPU model weights.

Programmatic use::

    from sheaf.openapi import generate
    from sheaf.spec import ModelSpec

    specs = [
        ModelSpec(name="chronos", model_type=ModelType.TIME_SERIES, backend="chronos2"),
        ModelSpec(name="flux", model_type=ModelType.DIFFUSION, backend="flux"),
    ]
    schema: dict = generate(specs)
    Path("openapi.json").write_text(json.dumps(schema, indent=2))

CLI use (run as a module)::

    python -m sheaf.openapi --specs my_specs:specs > openapi.json

where ``my_specs:specs`` is the dotted path to a Python module attribute
holding a ``list[ModelSpec]``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

from sheaf.modal_server import _build_asgi_app
from sheaf.spec import ModelSpec

__all__ = ["generate", "main"]


def generate(specs: list[ModelSpec] | None = None) -> dict[str, Any]:
    """Return the OpenAPI 3 schema for a sheaf deployment.

    Backends are not loaded — the schema only reflects route shape and
    Pydantic request/response types, both of which are static.

    Note that ``ModalServer`` registers templated routes (``/{name}/predict``,
    etc.) so the OpenAPI surface is the same regardless of which deployments
    are passed.  ``specs`` is accepted for API consistency with
    ``_build_asgi_app`` and to preserve forward-compat if per-deployment
    schema variation is added later.

    Args:
        specs: Optional list of ``ModelSpec``s.  Defaults to ``[]`` since
            the schema doesn't actually depend on the deployment list today.

    Returns:
        The FastAPI-generated OpenAPI dict.  Serialise with ``json.dumps``
        to write to disk.
    """
    app = _build_asgi_app(specs or [], load_backends=False)
    return app.openapi()


def _load_specs_from_dotted_path(path: str) -> list[ModelSpec]:
    """Resolve ``module.path:attr`` to a ``list[ModelSpec]``.

    Args:
        path: ``"my_pkg.my_module:specs"`` style attribute reference.

    Returns:
        The attribute value, validated as a list of ``ModelSpec``.

    Raises:
        ValueError: If the attribute is missing or isn't a list of ``ModelSpec``.
    """
    if ":" not in path:
        raise ValueError(f"--specs must be 'module.path:attribute', got {path!r}")
    module_path, attr = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, attr, None)
    if obj is None:
        raise ValueError(f"Attribute {attr!r} not found in module {module_path!r}")
    if not isinstance(obj, list) or not all(isinstance(s, ModelSpec) for s in obj):
        raise ValueError(f"{path} must be a list[ModelSpec]; got {type(obj).__name__}")
    return obj


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m sheaf.openapi``.

    Args:
        argv: Argument list (default ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="python -m sheaf.openapi",
        description="Generate the OpenAPI 3 schema for a sheaf deployment.",
    )
    parser.add_argument(
        "--specs",
        required=True,
        help=(
            "Dotted path to a list[ModelSpec], e.g. 'my_specs:specs'.  "
            "The module must be importable from the current Python path."
        ),
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output file path; '-' (default) writes to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level (default 2). Use 0 for one-line output.",
    )
    args = parser.parse_args(argv)

    try:
        specs = _load_specs_from_dotted_path(args.specs)
    except (ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    schema = generate(specs)
    rendered = json.dumps(schema, indent=args.indent or None)

    if args.out == "-":
        print(rendered)
    else:
        with open(args.out, "w") as f:
            f.write(rendered)
            f.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
