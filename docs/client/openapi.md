# OpenAPI export

Every sheaf-serve deployment serves a FastAPI app, so it exposes
`/openapi.json` automatically. For language-agnostic clients —
TypeScript, Go, Rust — generate from that spec with the OpenAPI
generator of your choice.

For ahead-of-time export (so non-Python teams can bootstrap a client
without spinning up a server first), use the bundled CLI:

```bash
python -m sheaf.openapi --specs my_module:specs > openapi.json
```

`my_module:specs` should reference an iterable of `ModelSpec`s in your
codebase:

```python title="my_module.py"
from sheaf import ModelSpec
from sheaf.api.base import ModelType

specs = [
    ModelSpec(
        name="chronos",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
    ),
    ModelSpec(
        name="embedder",
        model_type=ModelType.EMBEDDING,
        backend="openclip",
    ),
]
```

The generated spec includes:

- One operation per deployment route (`/{name}/predict`,
  `/{name}/health`, `/{name}/ready`, `/{name}/stream`)
- Schemas for every request and response variant in `AnyRequest` /
  `AnyResponse`, discriminated by `model_type`
- The schemas for `BaseRequest`, `BaseResponse`, and the per-model-type
  contracts

## Why a CLI export and not just hitting `/openapi.json`

Exporting at build time decouples the client codegen from the server
deployment cycle. CI can run `python -m sheaf.openapi` against the
specs file the team commits to git; the resulting spec changes in
source control are the contract review surface; non-Python teams
generate clients from a tagged commit, not a live URL that may be
ahead or behind.

## Programmatic API

```python
from sheaf import ModelSpec
from sheaf.openapi import generate

spec = ModelSpec(name="chronos", model_type="time_series", backend="chronos2")
schema = generate([spec])  # dict — write to JSON or hand to a generator directly
```

`generate()` builds a FastAPI app via `_build_asgi_app(specs,
load_backends=False)`, so no GPU / model weights / extras are needed
to produce the spec. Safe to run in CI from any environment that has
`sheaf-serve` core installed.

## Reference

The CLI lives at `python -m sheaf.openapi`; programmatic API is
`sheaf.openapi.generate(specs)`. Inspect the source at
[`src/sheaf/openapi.py`](https://github.com/korbonits/sheaf/blob/main/src/sheaf/openapi.py).
