# Contributing to Sheaf

Thanks for your interest. Sheaf is early — the best contributions right now are new backend implementations and feedback on the API contracts.

## Development setup

```bash
git clone https://github.com/korbonits/sheaf.git
cd sheaf
uv sync --extra dev
uv run pre-commit install   # wire up pre-commit hooks (required, per-clone)
uv run pytest tests/
```

## What to contribute

### New backends (highest value)

Each new model type needs:
1. A typed request/response contract in `src/sheaf/api/`
2. A `ModelBackend` implementation in `src/sheaf/backends/`
3. Tests in `tests/`

The time series contract (`src/sheaf/api/time_series.py`) and Chronos2 backend (`src/sheaf/backends/chronos.py`) are the reference implementation. Follow that pattern.

**Wanted backends (v0.5 targets, in priority order):**
- LiDAR / 3D point cloud (PointNet++, OpenShape)
- Pose estimation (ViTPose, MediaPipe)
- Optical flow (RAFT, UniMatch)
- Multimodal generation — text+image-conditioned (SDXL, CogVideoX)
- Speech synthesis with fine-grained control (StyleTTS2, Kokoro)

### API contract feedback

If you have strong opinions about what a request/response contract should look like for a new modality — open an issue. Getting the contracts right before building the optimizations behind them is the priority at this stage.

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

## Implemented backends (v0.4 — complete)

All backends below are implemented, tested, and wired into the Ray Serve smoke suite:

| Backend | Registry key | Install extra | Notes |
|---|---|---|---|
| Chronos2 | `chronos2` | `time-series` | Chronos-Bolt and Chronos-T5 families |
| TimesFM | `timesfm` | `time-series` | Google TimesFM |
| Moirai | `moirai` | `moirai` | Salesforce Moirai (uni2ts) |
| TabPFN | `tabpfn` | `tabular` | Classification + regression; requires `TABPFN_TOKEN` |
| Whisper | `whisper` | `audio` | openai-whisper (PyTorch) |
| FasterWhisper | `faster_whisper` | `audio` | CTranslate2; no torch at inference |
| MusicGen | `musicgen` | `audio-generation` | Meta MusicGen; text → audio |
| Bark | `bark` | `tts` | Suno Bark via transformers |
| OpenCLIP | `open_clip` | `vision` | Image and text embeddings |
| DINOv2 | `dinov2` | `vision` | Image embeddings; CLS or mean pooling |
| SAM2 | `sam2` | `vision` | Prompted segmentation |
| DepthAnything | `depth_anything` | `vision` | Monocular depth estimation |
| DETR | `detr` | `vision` | Object detection; any `AutoModelForObjectDetection` |
| ESM-3 | `esm3` | `molecular` | Protein embeddings; Python 3.12+ |
| NucleotideTransformer | `nucleotide_transformer` | `genomics` | DNA/RNA embeddings |
| MolFormer | `molformer` | `small-molecule` | Small molecule SMILES embeddings |
| MACE | `mace` | `materials` | Universal interatomic potential (energy, forces, stress) |
| Prithvi | `prithvi` | `earth-observation` | IBM/NASA geospatial embeddings |
| GraphCast | `graphcast` | `weather` | Google DeepMind weather forecasting |
| ImageBind | `imagebind` | `multimodal` | Cross-modal embeddings; install from source (not on PyPI) |
| FLUX | `flux` | `diffusion` | FLUX.1-schnell / FLUX.1-dev image generation |
| VideoMAE | `videomae` | `video` | VideoMAE / TimeSformer embeddings + Kinetics-400 classification |

## Feast feature store integration

`TimeSeriesRequest` accepts either raw `history` or a `feature_ref` — a pointer into a Feast online feature store. The serving layer resolves the feature before inference; backends always see `history`.

To use Feast, set `feast_repo_path` on `ModelSpec`:

```python
spec = ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    feast_repo_path="/feast/feature_repo",
)
```

Then send requests with `feature_ref` instead of `history`:

```json
{
    "model_type": "time_series",
    "model_name": "chronos",
    "feature_ref": {
        "feature_view": "asset_prices",
        "feature_name": "close_history_30d",
        "entity_key": "ticker",
        "entity_value": "AAPL"
    },
    "horizon": 7,
    "frequency": "1d"
}
```

See `examples/quickstart_feast.py` for a full end-to-end example with a local SQLite store.

## Modal serverless deployment

`ModalServer` is a zero-infra alternative to `ModelServer` (Ray Serve). It wraps the same `ModelSpec` list and deploys to Modal's managed GPU infrastructure:

```python
from sheaf import ModalServer

server = ModalServer(models=[spec], app_name="my-sheaf", gpu="A10G")
app = server.app  # modal deploy my_server.py
```

See `examples/quickstart_modal.py` for a full example.

## Code style

```bash
uv run ruff check src/ tests/         # lint (check)
uv run ruff check --fix src/ tests/   # lint (auto-fix)
uv run ruff format src/ tests/        # format
uv run ty check src/                  # type check
```

Pre-commit hooks run the same checks as CI (`ruff check`, `ruff format --check`, `ty check`) and block the commit if anything fails. Run `uv run pre-commit install` once after cloning to activate them.

## License

By contributing, you agree your contributions will be licensed under Apache 2.0.
