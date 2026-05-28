# ADR 0001 — ESMC + ESMFold2 integration, and the `structure` model category

**Status:** Accepted (draft PR)
**Date:** 2026-05-27
**Authors:** Sheaf maintainers

## Context

On 2026-05-27 Chan Zuckerberg Biohub released "a world model of protein
biology" — three artifacts: **ESMC** (protein language model), **ESMFold2**
(structure prediction model built on ESMC 6B), and **ESM Atlas** (a dataset of
6.8 B sequences / 1.1 B predicted structures). This ADR records the verification
work done before integrating ESMC and ESMFold2 as first-class Sheaf model
types, the design decisions taken in `sheaf.api.protein_language` /
`sheaf.api.structure`, and why ESMFold2 forces us to introduce a new
top-level model category (`STRUCTURE`).

ESM Atlas is a dataset, not a model, and is out of scope for this PR.

## Upstream verification (the "before any code" step)

### Repo and license

- `github.com/evolutionaryscale/esm` issues HTTP 301 → `github.com/Biohub/esm`
  (verified via `curl -sSI`; the repo has been re-homed under the Biohub
  GitHub org as of the release).
- `LICENSE.md` at `Biohub/esm@main` is a standard MIT license, Copyright
  2026 Chan Zuckerberg Biohub, Inc. (fetched verbatim via
  `raw.githubusercontent.com`). No additional clauses or carve-outs.
- The README's "Licenses" section says: "These models are available under
  the [MIT license](https://github.com/Biohub/esm/blob/main/LICENSE.md)."

### Python package

- Distribution: `pip install esm@git+https://github.com/Biohub/esm.git@c94ed8d`.
  A PyPI release is described as "coming soon" but is **not** yet published as
  of this PR. We pin to commit `c94ed8d` to stay reproducible; we will switch
  to a PyPI version constraint when one ships.
- Naming conflict: the new `esm` package shares the import name with the
  pre-2026 `esm` package (PyPI 3.x) used by our existing `ESM3Backend`
  (`sheaf.backends.esm3`, extra `[molecular]`). Both cannot be installed in
  the same environment under the same import name. We treat this as a
  packaging constraint, declared via `[tool.uv]` `conflicts`. See
  "Compatibility with existing `[molecular]` extra" below.

### Model artifacts (HuggingFace)

| Artifact                 | HF repo ID            | Access            | Use via                                |
|--------------------------|-----------------------|-------------------|----------------------------------------|
| ESMC 6B                  | `Biohub/ESMC-6B`      | Weight download   | `transformers.AutoModelForMaskedLM`    |
| ESMFold2                 | `biohub/ESMFold2`     | Weight download   | `transformers.models.esmfold2.modeling_esmfold2.ESMFold2Model` + `esm.models.esmfold2.ESMFold2InputBuilder` |
| ESMC 300M / 600M         | `esmc-{300m,600m}-2024-12` | **Forge API-only** | `esm.sdk.esmc_client(...)` with a Biohub Platform token |
| ESMFold2 fast            | `esmfold2-fast-2026-05` | **Forge API-only** | `esm.sdk.forge.SequenceStructureForgeInferenceClient` |

Note the case inconsistency in the README itself (`Biohub/ESMC-6B` vs
`biohub/ESMFold2`); we mirror the upstream strings verbatim rather than
normalising.

We could **not** independently fetch the HF model card pages from this
build environment (`huggingface.co` is not in the outbound allowlist:
`x-deny-reason: host_not_allowed`). We are relying on the README in the
Biohub/esm repository — which is fetchable — for the canonical
`from_pretrained` strings and license restatement. The README explicitly
restates MIT, so we treat this as verified.

### ESMFold2 inference-time scaling parameters

Per the README's local-inference example, `ESMFold2InputBuilder().fold(...)`
exposes four scaling parameters as positional kwargs:

```python
result = ESMFold2InputBuilder().fold(
    model, spi,
    num_loops=3,            # depth of the looped-transformer recurrence
    num_sampling_steps=50,  # diffusion steps per sample
    num_diffusion_samples=1,
    seed=0,
)
```

Result fields used in this PR: `result.plddt` (per-residue confidence),
`result.ptm`, `result.iptm`, `result.complex.to_mmcif()` /
`result.complex.to_pdb()`. PAE matrix and per-sample ranking scores are
documented as outputs in the paper but are not shown in the README sample;
we leave the `pae` and `sample_scores` response fields optional and populate
them only when present on the result object.

**pLDDT scale**: empirically (verified by a Modal H100 smoke against the
default `biohub/ESMFold2` weights, 2026-05-27), ESMFold2 returns `plddt` as
a `torch.Tensor` of fractional values on `[0, 1]` — *not* the conventional
AlphaFold / ESMFold-v1 `[0, 100]` scale. We pass through faithfully (no
backend-side scaling) and document the scale on `StructureResponse.plddt`
so callers can multiply by 100 themselves if they want the conventional
values. Faithful pass-through is consistent with Sheaf's general "validate
at the boundary, don't transform inside backends" convention.

## Decision

### 1. Two new model categories

We add two new `ModelType` enum values:

- `PROTEIN_LANGUAGE = "protein_language"` — for ESMC and any future protein
  language model that returns per-token logits / per-token embeddings (not
  pooled to one vector per sequence).
- `STRUCTURE = "structure"` — for protein structure prediction. This is a
  new top-level model category for Sheaf. Structure prediction does not
  fit any existing category: it produces a 3D structure (atom coordinates),
  not an embedding, classification, generation, or forecast. It also has
  fundamentally different batching and caching properties (see below).

We deliberately keep `MOLECULAR` (per-sequence pooled embeddings, ESM-3)
distinct from `PROTEIN_LANGUAGE` (per-token logits + embeddings, ESMC).
The response shapes are incompatible (`(N, D)` vs `(N, L, V)`), and trying
to unify them would force every caller to handle both shapes.

### 2. Two new API contracts

- `sheaf.api.protein_language.ProteinLanguageRequest / Response` — ESMC
  contract. Per-sequence ragged outputs: per-token logits and optional
  per-token embeddings, with `seq_lens` so callers can slice the padded
  output back to per-sequence length.
- `sheaf.api.structure.StructureRequest / Response` — ESMFold2 contract.
  Multi-chain input (`list[ChainInput]`), inference-time scaling
  parameters as first-class fields (`num_loops`, `num_sampling_steps`,
  `num_samples`, `seed`), and structure output in PDB or mmCIF.

### 3. Two new backends

- `sheaf.backends.esmc.ESMCBackend` (registered as `"esmc"`).
- `sheaf.backends.esmfold2.ESMFold2Backend` (registered as `"esmfold2"`).

Both follow the existing CLIP / DINOv2 / ESM-3 / MolFormer convention:
lazy imports inside `load()`, `_tokenizer` / `_model` / `_Image`-style
instance attributes stored at `load()` for test injectability, no
heavyweight imports at module level.

### 4. New `[protein]` extra

`pyproject.toml` gets a new optional-dependency group:

```toml
protein = [
    "esm @ git+https://github.com/Biohub/esm.git@c94ed8d ; python_full_version >= '3.12'",
    "transformers>=4.40.0",
    "torch>=2.0.0",
]
```

Pinned to Python 3.12+ to match the upstream package's `requires-python`.
Conflicts with the existing `[molecular]` extra (same `esm` import name,
different versions) — declared in `[tool.uv]` `conflicts`. Users who need
both ESM-3 and ESMC will need to run them in separate Sheaf deployments.

We do **not** remove or migrate the existing `[molecular]` extra. ESM-3
remains available; the two extras are mutually exclusive (uv enforces this).
When the new `esm` package ships to PyPI and we have confirmed it supports
ESM-3 inference, a follow-up PR can consolidate the extras and drop the
conflict declaration.

### 5. Forge / API-only variants are out of scope for this PR

ESMC 300M / 600M and ESMFold2-fast require a Biohub Platform API token
and the `esm.sdk.esmc_client` / `SequenceStructureForgeInferenceClient`
classes. They live behind a different code path (HTTP client, no local
weights). Adding them is a strictly additive change once a user explicitly
asks for it; for v0.11 we ship only the weight-downloadable variants:

| Sheaf backend | model_name                | requires_forge |
|---------------|---------------------------|----------------|
| `esmc`        | `Biohub/ESMC-6B` (default)| no             |
| `esmfold2`    | `biohub/ESMFold2`         | no             |

The `ESMCBackend` constructor accepts an arbitrary `model_name` and a
`requires_forge: bool` flag plumbed through to the load path. When
`requires_forge=True` we raise a `NotImplementedError` immediately with a
pointer to this ADR — better than a silent fall-through that 404s on HF
download. Wiring up the Forge HTTP client is a deliberate future PR.

### 6. Batching, caching, streaming

- **ESMC batching**: tokenize the full `request.sequences` list with
  `padding=True` and run a single forward pass — same as `MolFormerBackend`.
  Variable-length sequences are handled via the attention mask. Sheaf's
  `BatchPolicy.bucket_by` can group requests by sequence length if a
  caller sets `bucket_by="max_len"` on the deployment; we expose
  `max_len` as a derived field on the response so length-bucketing
  works out of the box.
- **ESMFold2 batching**: per-request compute varies hugely with sequence
  length and `num_samples`. The upstream `ESMFold2InputBuilder().fold()`
  is single-sample-per-call. We do not implement a true batched forward;
  `batch_predict` runs requests sequentially. Sheaf operators who care
  about throughput should size the Ray Serve replica count to handle
  expected concurrency (one in-flight prediction per replica).
- **Caching**: ESMC plugs into the existing `ResponseCache` via
  `CacheConfig(enabled=True)` with no changes — the existing key
  derivation (`request.model_dump(mode="json", exclude={"request_id"})`
  → SHA-256) already includes the sequence string and the
  `return_logits` / `return_embeddings` flags. For ESMFold2 we
  recommend `CacheConfig(exclude_fields=["seed"])` only if callers want
  same-sequence cache hits across different random seeds — by default
  seeds participate in the key, matching diffusion-model behaviour
  established in v0.7.
- **Streaming**: neither backend supports the `stream_predict` path in
  v0.11. ESMFold2's diffusion loop could plausibly emit per-step events
  in a future PR (mirroring `FluxBackend.stream_predict`), but the
  upstream API does not currently expose a step-end callback.

### 7. Observability

ESMC and ESMFold2 use the existing `sheaf.metrics.record_predict(...)`
and `sheaf.tracing.trace_predict(...)` paths. No new spans, no new metric
families. The deployment `name` label (e.g. `"esmc-6b"`) distinguishes
backends on dashboards. We chose this over the task spec's
`sheaf.esmc.forward` / `sheaf.esmfold2.forward` span names because Sheaf's
convention is one canonical predict span per request with the deployment
label as the disambiguator — adding per-backend span names would create
inconsistent telemetry across the 25+ existing backends.

## Consequences

- Sheaf now serves 25 model types across 22 model categories (+2 from
  this PR: `PROTEIN_LANGUAGE`, `STRUCTURE`).
- The `[molecular]` and `[protein]` extras are mutually exclusive at
  install time. Documentation needs to flag this.
- The pinned `esm @ git+...@c94ed8d` will become stale; we should track a
  PyPI release in the Biohub repo and switch when one lands.
- `STRUCTURE` is the first model category whose output is fundamentally
  non-tensor (PDB / mmCIF strings, with structured side-channel data
  like pLDDT and PAE). Future structure-prediction backends (Boltz-1,
  Chai-1, etc.) can reuse the contract without new infra.
- ESMFold2 outputs can be large (mmCIF for a multi-chain complex is
  often >100 KB); the existing `ResponseCache` is in-process LRU, which
  is fine for small deployments but won't scale to thousands of cached
  predictions per replica. A pluggable disk-backed cache is a future
  enhancement, not blocking for v0.11.

## Alternatives considered

- **Reuse `MOLECULAR` for ESMC.** Rejected — response shapes are
  incompatible (pooled vector vs ragged per-token tensor), and forcing
  callers to branch on `model_name` defeats the point of a typed
  contract.
- **Bundle structure prediction under `MOLECULAR`.** Rejected — pLDDT,
  PAE, multi-chain inputs, and a PDB string output have nothing in
  common with the embedding contract.
- **Default ESMC to a Forge variant for parity with the README's
  300M/600M code samples.** Rejected — Forge requires an API token and
  network access; weight-downloadable inference is the model surface
  Sheaf can serve self-hosted today.
- **Drop ESM-3 in favour of the new ESMC package.** Rejected for this
  PR — the new package may support ESM-3 (the README references an
  `ESM3_README.md`), but we haven't verified end-to-end equivalence. A
  follow-up PR can consolidate once tested.

## References

- Biohub/esm README: <https://github.com/Biohub/esm>
- ESM Atlas: <https://biohub.ai/esm/protein/atlas>
- Preprint: "Language Modeling Materializes a World Model of Protein
  Biology" (Candido et al., 2026), <https://biohub.ai/papers/esm_protein.pdf>
- HF collections (not directly fetched from build env — see
  "verification" above):
  - <https://huggingface.co/collections/Biohub/esmc-model-family>
  - <https://huggingface.co/collections/biohub/esmfold2-model-family>
