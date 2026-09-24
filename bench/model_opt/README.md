# Inference optimization kit benchmarks

Numbers for `ModelSpec.model_opt` (see `docs/concepts/model_opt.md`).

## ESM C (`off` vs `exact`)

```bash
modal run bench/model_opt/modal_esmc.py::fetch_weights   # once: pinned 6B snapshot (~25 GB) into a volume
modal run bench/model_opt/modal_esmc.py::gpu_test        # exact == off, bit for bit
modal run bench/model_opt/modal_esmc.py::bench           # writes bench/results/<date>-esmc-kit-h100-6b/
```

Or on any GPU host inside the kit image
(`examples/docker/model-opt/build_esmc.sh`):

```bash
python bench/model_opt/bench_esmc.py --model esmc_6b --out-dir bench/results/esmc-kit-local
```

What is measured, and why:

- **One mode per process.** The kit patches the interpreter process-wide, so
  the driver runs each mode in a fresh subprocess.
- **`forward` vs `predict`.** `forward` is the model call alone, where the
  kit acts. `predict` is `backend.predict` end to end, including converting
  logits to JSON-able lists. That conversion costs the same in both modes, so
  the served speedup is smaller than the forward speedup. Both are reported.
- **Grid.** `--lengths 64 128 256 512 1024 2048` × `--batches 1 8 32` by
  default. Sequences are seeded random strings of the 20 standard amino
  acids. Out-of-memory cells are recorded as `oom`, not dropped.
- **Memory.** `torch.cuda.max_memory_allocated` / `max_memory_reserved` per
  cell.
- **Environment.** GPU model, driver (`nvidia-smi`), CUDA, cuDNN, torch,
  triton, flash-attn, Transformer Engine, `esm` (+ git commit), the
  transformers fork commit, `esmc_opt`, the kits tree ref and Sheaf's SHA are
  written into every result.
- **Load time.** Reported separately (first `exact` load also compiles; the
  compile cache on the `/jit` volume makes later loads warm).
