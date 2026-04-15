# Contributing to Sheaf

Thanks for your interest. Sheaf is early — the best contributions right now are new backend implementations and feedback on the API contracts.

## Development setup

```bash
git clone https://github.com/korbonits/sheaf.git
cd sheaf
uv sync --extra dev
uv run pytest tests/
```

## What to contribute

### New backends (highest value)

Each new model type needs:
1. A typed request/response contract in `src/sheaf/api/`
2. A `ModelBackend` implementation in `src/sheaf/backends/`
3. Tests in `tests/`

The time series contract (`src/sheaf/api/time_series.py`) and Chronos2 backend (`src/sheaf/backends/chronos.py`) are the reference implementation. Follow that pattern.

**Wanted backends (in priority order):**
- TabPFN (`tabular`)
- TimesFM (`time_series`)
- ESM-3 (`molecular`)
- Whisper (`audio`)
- GraphCast (`geospatial`)

### API contract feedback

If you have strong opinions about what a tabular or molecular request should look like — open an issue. Getting the contracts right before building the optimizations behind them is the priority at this stage.

### Bug reports

Please include:
- Python version
- `sheaf-serve` version (`pip show sheaf-serve`)
- Minimal reproducer

## Adding a backend

Register your backend with the `@register_backend` decorator:

```python
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

@register_backend("my-model")
class MyModelBackend(ModelBackend):
    def __init__(self, model_id: str, **kwargs) -> None:
        self._model_id = model_id
        self._model = None

    @property
    def model_type(self) -> str:
        return "time_series"  # or whichever ModelType applies

    def load(self) -> None:
        # Load weights here — called once at server startup
        self._model = load_my_model(self._model_id)

    def predict(self, request) -> BaseResponse:
        # Single request inference
        ...

    def batch_predict(self, requests) -> list[BaseResponse]:
        # Override for model-type-aware batching
        # Default: runs predict() sequentially
        ...
```

Then add your model's optional dependencies to `pyproject.toml` under `[project.optional-dependencies]`.

## Code style

```bash
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/sheaf             # type check
```

CI enforces lint and format on every PR.

## License

By contributing, you agree your contributions will be licensed under Apache 2.0.
