# Models

sheaf-serve ships ready-to-use backends for 20+ model types across
seven domains. Each backend lives behind an extras flag — install
only what you need.

## Catalogue

| Domain | Model | Backend (`backend=`) | Extras flag |
|---|---|---|---|
| **Time series** | Chronos2 / Chronos-Bolt | `chronos2` | `time-series` |
| | TimesFM v2 | `timesfm` | `time-series` |
| | Moirai | `moirai` | `moirai` |
| **Tabular** | TabPFN v2 | `tabpfn` | `tabular` |
| **Audio (ASR)** | Whisper | `whisper` | `audio` |
| | faster-whisper | `faster-whisper` | `audio` |
| **Audio generation** | MusicGen | `musicgen` | `audio-generation` |
| **Text-to-speech** | Bark | `bark` | `tts` |
| | Kokoro | `kokoro` | `kokoro` |
| **Vision (embedding)** | OpenCLIP | `openclip` | `vision` |
| | DINOv2 | `dinov2` | `vision` |
| **Vision (segmentation)** | SAM2 | `sam2` | `vision` |
| **Vision (depth)** | Depth Anything v2 | `depth-anything` | `vision` |
| **Vision (detection)** | DETR / RT-DETR | `detr` | `vision` |
| **Vision (pose)** | ViTPose | `vitpose` | `pose` |
| **Vision (optical flow)** | RAFT | `raft` | `optical-flow` |
| **Vision (video)** | VideoMAE / TimeSformer | `videomae` | `video` |
| **Image diffusion** | FLUX (schnell / dev) | `flux` | `diffusion` |
| **Multimodal generation** | SDXL img2img / inpaint | `sdxl` | `multimodal-generation` |
| **Cross-modal embedding** | ImageBind | `imagebind` | `multimodal` (+ git install) |
| **Molecular (protein)** | ESM-3 | `esm3` | `molecular` (Python 3.12+) |
| **Genomics** | Nucleotide Transformer | `nucleotide-transformer` | `genomics` |
| **Small molecule** | MolFormer | `molformer` | `small-molecule` |
| **Materials** | MACE-MP-0 | `mace` | `materials` |
| **Weather** | GraphCast | `graphcast` | `weather` |
| **Earth observation** | Prithvi-EO | `prithvi` | `earth-observation` |
| **LiDAR / 3D** | PointNet | `pointnet` | `lidar` |

## Cross-cutting extras

| Extras flag | What it adds |
|---|---|
| `feast` | Feast online feature store integration ([Feast →](../concepts/feast.md)) |
| `modal` | Modal serverless deployment via `ModalServer` ([Modal →](../deployment/modal.md)) |
| `metrics` | Prometheus exporter (`/metrics` endpoint) |
| `tracing` | OpenTelemetry SDK + OTLP exporter |
| `batch` | Offline batch inference via Ray Data ([Batch →](../concepts/batch.md)) |
| `worker` | Async-job worker (Redis Streams) ([Worker →](../concepts/worker.md)) |
| `all` | Everything except narrow-Python or git-only extras |

## Usage shape

Every model type follows the same shape:

```python
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType

server = ModelServer(models=[
    ModelSpec(
        name="<your-name>",
        model_type=ModelType.<TYPE>,
        backend="<backend-name-from-table>",
        backend_kwargs={...},
    ),
])
server.run()
```

The typed request and response live in `sheaf.api.<module>` —
`TimeSeriesRequest` in `sheaf.api.time_series`, `EmbeddingRequest` in
`sheaf.api.embedding`, etc. Inspect a backend's request schema with
the OpenAPI export ([OpenAPI →](../client/openapi.md)).

## Quickstart examples

The repo ships a runnable `examples/quickstart_<type>.py` for every
major category — go to
[`examples/`](https://github.com/korbonits/sheaf/tree/main/examples) on GitHub.

## Adding a new backend

If sheaf-serve doesn't ship the backend you need, the plugin model is
deliberately small: subclass `ModelBackend`, decorate with
`@register_backend("name")`, and reference it by name from
`ModelSpec.backend`. Or pass the class directly via
`ModelSpec.backend_cls` for one-off / private backends. See
[CONTRIBUTING.md](https://github.com/korbonits/sheaf/blob/main/CONTRIBUTING.md)
for the full walk-through.
