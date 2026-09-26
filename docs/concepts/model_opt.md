# Inference optimization kits

Sheaf can serve some models through Anthropic's open-sourced biomolecular
inference optimization kits
([uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling),
Apache-2.0). A kit makes the upstream model faster (or lighter on memory)
without changing how it is called. You opt in per deployment:

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.model_opt import ModelOptConfig
from sheaf.spec import ResourceConfig

spec = ModelSpec(
    name="esmc-6b",
    model_type=ModelType.PROTEIN_LANGUAGE,
    backend="esmc",
    backend_kwargs={"model_name": "esmc_6b", "device": "cuda"},
    resources=ResourceConfig(num_gpus=1),
    model_opt=ModelOptConfig(mode="exact", jit_root="/var/cache/sheaf/model_opt_jit"),
)
```

`model_opt` is `None` by default. With it unset, nothing of any kit is
imported and every backend runs exactly the code path it ran before.

## Modes

One vocabulary across all kits; each backend accepts the subset its kit ships.

| mode | outputs | use it for |
|---|---|---|
| `off` | the pinned upstream release exactly as published | the reference, and A/B baselines |
| `exact` | **bitwise identical** to `off`, faster | production when outputs must not move |
| `fast` | small, documented numeric differences, faster still | throughput — **not** bit-identical |
| `big` | as `fast`, lowest peak GPU memory | inputs the other modes cannot hold |

| backend | kit | modes | notes |
|---|---|---|---|
| `esmc` | `esmc` (ESM C, `esm` 3.4.0) | `off`, `exact` | 300M / 600M / 6B |
| `esmfold2` | `esmfold2` (ESMFold2, `esm` 3.3.0) | `off`, `exact`, `fast` | `biohub/ESMFold2` (kit variant `full_nomsa`), `biohub/ESMFold2-Fast` (`fast`) |

`ModelOptConfig` fields:

- `mode` — as above. An unsupported mode is rejected at deploy time.
- `jit_root` — persistent compile-cache root (see below).
- `levers_off` — switch named optimizations off (`MODEL_OPT_LEVERS_OFF`;
  names as on the kit's `LEVER` lines). The ESMFold2 kit accepts it; the ESM C
  kit has no such switch, so it must stay empty there.

### What `off` means for ESMC

With `model_opt` set, the ESMC backend loads through `esm`'s own client
(`esm.models.esmc.ESMC.from_pretrained`: bf16 on GPU, the shipped
flash-attention class) and runs the padded batched forward under bf16
autocast. That is the kit's stock reference, and `exact` is byte-identical to
it for the same inputs and batch composition. With `model_opt=None` the
backend keeps its original `transformers.AutoModelForMaskedLM` path, which is
a different numerical configuration. Compare `exact` with `off`, never with
`None`.

### What `off` means for ESMFold2

`off` is the library's fused backend with pair-block chunking off
(`set_kernel_backend("fused")` + `set_chunk_size(None)`, the kit's
`--backend fused`). That is the configuration `exact` reproduces, bit for bit
under the kit's deterministic recipe: `CUBLAS_WORKSPACE_CONFIG=:4096:8` set
before torch is imported, `torch.use_deterministic_algorithms(True,
warn_only=True)`, and a seeded request. Without the recipe, stock itself is not
run-to-run reproducible, so neither is `exact`. `fast` stays within stock's
seed-to-seed variation and is not bitwise. With `model_opt=None` the backend
runs the model as `from_pretrained` loads it, on Sheaf's `[protein]` pins.

The kit runs one variant per process (`biohub/ESMFold2` → `full_nomsa`,
`biohub/ESMFold2-Fast` → `fast`), and Sheaf enforces that alongside the mode.

## Startup contract

A kit mode either engages or refuses by name. Sheaf activates the kit
explicitly inside the backend's `load()`, runs one short warm-up call, and
checks the kit's own report. Any refusal raises `ModelOptNotActiveError`, so
the deployment fails **at startup** (Ray Serve marks it `DEPLOY_FAILED`;
a `ModalServer` container fails while building its app) with the kit's reason. It
never fails on the first request, and it never falls back to stock silently.

The kit's own lines, `[esmc-opt] ACTIVE mode=exact …` and
`[esmc-opt] APPLIED … levers_applied=pipe,fused levers_fallback=none …`, are
re-logged through the `sheaf.model_opt` logger (JSON with
`SHEAF_LOG_JSON=1`, with `kit` and `deployment` fields). They are also kept on
the backend as `backend.model_opt_lines`.

For ESMFold2 the kit applies its optimizations to the loaded model at load
(`esmfold2_opt.stack.apply_to`) and the check is its `partial` list: a lever of
the mode left unapplied refuses the deployment. When flash-attention or
Transformer Engine is not live, the kit exits the process (code 3); Sheaf turns
that into `ModelOptNotActiveError` too, so a Ray replica fails its startup
instead of dying.

Sheaf does not use the kits' `<KIT>_OPT` environment variables. That route
`os._exit(3)`s the interpreter on refusal, which reaches Ray only as an
opaque "actor died". Leave `ESMC_OPT` unset in Sheaf containers.

## One mode per process

A kit patches the upstream package in memory for the whole interpreter. The
ESM C kit, for example, patches every ESMC client built while it is active.
Sheaf enforces one kit mode per process:

- **Ray Serve.** Each replica is its own process, so different deployments
  can run different modes on the same node.
- **`ModalServer`.** All specs share one container, so a spec with
  `model_opt` must be the only spec of its `ModalServer`. Run one app per
  mode.
- **`BatchRunner`.** `model_opt` requires `compute="actors"`, because task
  mode reuses worker processes across specs.
- **`SheafWorker`.** One spec per process already.

A second activation with a different mode in the same process raises
`ModelOptNotActiveError` naming both deployments.

## Environments and images

A kit only engages on its exact pinned stack. For the ESM C kit that means:
Python 3.12, CUDA 13.0 (driver ≥ 580), torch 2.11.0+cu130, triton 3.6.0,
`esm` 3.4.0 @ `43ccece2`, the Biohub `transformers` fork @ `ef32577f`,
flash-attn 2.7.4.post1, Transformer Engine 2.15.0, no xformers. The kit's pin
check refuses any other `esm` commit.

These pins cannot live in a PyPI extra (PyPI rejects git dependencies), and
they differ from `[protein]`'s `esm` @ `81b3646c`. Use the kit images
instead:

- **Docker / Ray Serve / KubeRay:** `bash examples/docker/model-opt/build_esmc.sh`
  builds the kit's own image unchanged, then installs Sheaf on top under a
  constraints file generated from that image. Pip can add Sheaf's dependencies
  but cannot move a pinned package, and the kit's pin check runs last.
- **Modal:** `bench/model_opt/modal_esmc.py` mirrors the kit's Dockerfile as a
  Modal image.

To try the kit on Modal without Sheaf, see
[`esmc_kit_exact.py`](https://github.com/korbonits/modal-examples/blob/9478f78d4df4cafb28c5e6fa7c7315fe354653d4/misc/esmc_kit_exact.py).
It is a single-file example that builds the same image, runs `off` and `exact`
in separate containers, compares their outputs byte for byte, and times both.

Weights are the kit's pinned Hugging Face snapshot. Fetch them once with
`run.sh install --weights DIR --variant 6b` (`modal run
bench/model_opt/modal_esmc.py::fetch_weights`), then serve with
`HF_HOME=DIR/hf HF_HUB_OFFLINE=1`.

The ESMFold2 kit has its own stack, which cannot share an environment with
ESM C's: Python 3.12, CUDA 13.0, torch 2.13.0+cu130, triton 3.7.1, `esm` 3.3.0
@ `26b0bc2b`, flash-attn 2.8.3.post1, Transformer Engine 2.15.0, xformers
0.0.35, plus the kits' shared core (`common/opt_core`). No index carries the
three CUDA extensions for this torch, so the image compiles them (the kit's
`environment/build_wheels.sh`, about 16 minutes on 24 cores).
`bench/model_opt/modal_esmfold2.py` builds that image on Modal (the extension
step gets 32 CPUs), fetches the three pinned snapshots (`biohub/ESMFold2`,
`biohub/ESMFold2-Fast`, `biohub/ESMC-6B`, ~27 GB) with the kit's installer, and
runs the GPU test and the benchmark. With `jit_root` set, Sheaf also points
the kit's weights-digest memo at `<jit_root>/weights`, so a restart does not
re-hash the checkpoints.

## Compile cache

Kernels compile (Triton) or build (CUDA extensions) on first use. Set
`jit_root` and Sheaf exports, unless already set:

- `TRITON_CACHE_DIR=<jit_root>/<stack key>/triton`
- `TORCH_EXTENSIONS_DIR=<jit_root>/<stack key>/torch_extensions`

The stack key is `torch<ver>-cu<cuda>-sm<cc>` (e.g. `torch2.11.0-cu130-sm90`),
the same layout the kits' `run.sh` uses. One root can therefore serve several
kits and cards.

| where | recommended `jit_root` |
|---|---|
| Docker image | bake it in: the ESM C kit's CUDA extension is built at image build (`/root/.cache/esmc_sdkfused`); a pre-filled cache tar can be added via the kit's `_jitcache/` build input |
| Modal | a `modal.Volume` mounted at `/jit` (`sheaf-model-opt-jit`); files are content-addressed, so concurrent containers are safe |
| Ray / KubeRay | a node-local directory, e.g. `/var/cache/sheaf/model_opt_jit` (hostPath) |

The kits refuse a cache directory that another user owns, that group or
others can write to, or that is a symbolic link. Sheaf creates missing
directories with mode 0700. Avoid shared NFS: root-squash and group-writable
mounts get refused, and Triton's file locking is unreliable there.

## Hardware caveat

The kits are tuned and stated for the **NVIDIA H100 80 GB** on Linux x86-64.
Other cards need the kit's per-card configuration (`configs/<card>.env` in
most kits). The kit reads the card at start-up; an optimization without a
kernel table for that card is either left out and named on the `ACTIVE` line,
or the mode refuses. For ESM C specifically, nothing reads the card name, so
H200 and A100 (80 / 40 GB) run the same code path. B200 (sm100) is untested
upstream: the pinned flash-attn / Transformer Engine wheels target sm80–sm90.

Benchmark numbers in `bench/results/*-esmc-kit-*` are H100 numbers. Do not
extrapolate them to other cards.

## Trust model

The kits are research performance tooling. They assume **trusted inputs,
trusted weights, and a single-user machine or container**. Before exposing a
kit mode to untrusted callers, note:

- They run inside, and with the privileges of, the upstream packages. Track
  `esm`'s and the transformers fork's advisories.
- Installing a kit adds a `.pth` start-up hook to the environment's
  `site-packages`. It is inert unless `<KIT>_OPT` is set, but every Python
  process in that environment imports it.
- Checkpoints are deserialized by the upstream loaders. Use only the kit's
  digest-pinned weights (`run.sh install --weights`), and keep
  `HF_HUB_OFFLINE=1` so a newer `main` is never pulled.
- Compile caches are digest-checked by the kits, but only when they sit in a
  private directory. Never point `jit_root` at a location other users can
  write.
- The kit images run as root, like the kits' own images. Use your platform's
  usual isolation (rootless runtime, no extra capabilities, minimal mounts).

Sheaf's request validation still runs at the boundary (Pydantic contracts),
but it does not make a kit safe against adversarial inputs.

## Attribution

The kits are Copyright 2026 Anthropic, PBC, under Apache-2.0. Each kit's
`stock/` carries the upstream project under its own licence. Sheaf does not
vendor them: images install them from a pinned commit and carry their
`LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` under
`/usr/share/doc/sheaf-model-opt/`. See `NOTICE` and `third_party/README.md`
in the repository.
