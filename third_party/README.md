# Third-party components

Nothing in this directory is vendored code. It records the external trees
Sheaf builds against, where they are pinned, and the attribution obligations
that come with them.

## Inference optimization kits

| | |
|---|---|
| Upstream | https://github.com/anthropics/uplifting-biomolecular-modeling |
| Status upstream | reference release; not maintained, no PRs (not archived on GitHub) |
| Licence | Apache-2.0 (the kits' original code); each kit's `stock/` is the upstream project under its own licence (see the tree's `NOTICE` and each kit's `THIRD_PARTY_NOTICES.md`) |
| Copyright | Copyright 2026 Anthropic, PBC |
| Pinned commit | `f4f62fa6592ae4938d49b1757bea0cfeff9f468e` (2026-09-17) |
| Kits used | `esmc` (ESM C: `off`, `exact`) |

**How Sheaf consumes it.** Not vendored. `common/opt_core` alone is ~306 MB
including prebuilt `.so` / `.cubin` binaries, and the ESMFold2 kit must be
installed editable (it locates its `forward/` tree next to the package), so
the kits are installed from a git checkout at the pinned commit inside the
kit images:

- `examples/docker/model-opt/build_esmc.sh` (Docker / Ray Serve / KubeRay)
- `bench/model_opt/modal_esmc.py` (Modal)

Both read `SHEAF_KITS_REPO` / `SHEAF_KITS_REF`; they default to the upstream
repository at the commit above. The plan is a Sheaf-maintained fork
(`korbonits/uplifting-biomolecular-modeling`) pinned at the same commit, so
the source cannot disappear; switching is a change of those two defaults.

**Attribution.** Apache-2.0 §4 applies to every image that redistributes a
kit. The Docker build copies the tree's `LICENSE` and `NOTICE` and the kit's
`LICENSE`, `THIRD_PARTY_NOTICES.md` and `third_party_licenses/` into
`/usr/share/doc/sheaf-model-opt/`; the kit's own directory in the image
(`/kit/esmc`) keeps them as shipped. Sheaf's top-level `NOTICE` names the kits.
If you vendor or modify kit files, keep their headers, mark your changes, and
carry the notices above.

**Security.** The kits assume trusted inputs and trusted weights on a
single-user machine or container (the tree's README, "Security
considerations"). Read `docs/concepts/model_opt.md` before exposing a kit mode
to untrusted callers.
