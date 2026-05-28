"""ESMFold2 on Modal — GPU structure prediction without a local GPU.

Drives ``sheaf.backends.esmfold2.ESMFold2Backend`` against an H100 on Modal,
exercising the full sheaf wrapper (load + predict + StructureResponse) rather
than the raw upstream API. Adapted from Modal's official ESMFold2 example
(modal-labs/modal-examples 06_gpu_and_ml/protein-folding/esmfold2.py), with
the same ``esm`` git revision pin for reproducibility.

Prerequisites:
    pip install modal
    modal setup              # authenticate once

Run:
    modal run examples/quickstart_protein_modal.py
    modal run examples/quickstart_protein_modal.py --sequence MKTAYIAK...

The model is cached in a persistent Modal volume after the first run; the
weight download (~12 GB ESMC backbone + ESMFold2 head) only happens once.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Pinned to the same upstream esm commit Modal's official example uses, so this
# smoke is reproducible against a known-working API surface.
ESM_REVISION = "81b3646c9429ea8458918415ad6a46178cb59833"

MINUTES = 60

app = modal.App(name="sheaf-esmfold2")

# Persistent HF cache volume — shared with Modal's reference example name so a
# user who has already run that pays no extra cold-start download cost.
esmfold2_volume = modal.Volume.from_name("esmfold2-models", create_if_missing=True)
models_dir = Path("/models")

# Image: Debian + git + uv-installed esm (git pin) + the sheaf-serve [protein]
# extra (transformers + torch). Mount the working src/ tree so we exercise the
# in-repo backend code, not a published PyPI release.
esmfold2_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        f"esm @ git+https://github.com/Biohub/esm.git@{ESM_REVISION}",
    )
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["protein"])
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env(
        {
            "HF_HOME": str(models_dir),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONPATH": "/root/src",
        }
    )
)


@app.cls(
    image=esmfold2_image,
    volumes={models_dir: esmfold2_volume},
    gpu="H100",
    timeout=20 * MINUTES,
)
class ESMFold2Inference:
    @modal.enter()
    def load_model(self) -> None:
        from sheaf.backends.esmfold2 import ESMFold2Backend

        print("loading ESMFold2 onto GPU via sheaf.backends.esmfold2")
        self.backend = ESMFold2Backend(model_name="biohub/ESMFold2", device="cuda")
        self.backend.load()
        print("ready")

    @modal.method()
    def fold(
        self,
        sequence: str,
        num_loops: int = 3,
        num_sampling_steps: int = 50,
        num_diffusion_samples: int = 1,
        seed: int = 0,
        output_format: str = "mmcif",
    ) -> dict:
        from sheaf.api.structure import ChainInput, StructureRequest

        req = StructureRequest(
            model_name="esmfold2",
            chains=[ChainInput(chain_id="A", sequence=sequence.strip())],
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_samples=num_diffusion_samples,
            seed=seed,
            output_format=output_format,  # type: ignore[arg-type]
        )
        print(
            f"folding len={len(sequence)} loops={num_loops} "
            f"steps={num_sampling_steps} samples={num_diffusion_samples} seed={seed}"
        )
        resp = self.backend.predict(req)
        return resp.model_dump(mode="json")


# Short test target from the PR's release checklist (~52 residues).
DEFAULT_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"


@app.local_entrypoint()
def main(
    sequence: str | None = None,
    output_path: str | None = None,
) -> None:
    seq = sequence or DEFAULT_SEQUENCE

    print(f"sheaf ESMFold2 smoke — sequence length {len(seq)}")
    inference = ESMFold2Inference()
    resp = inference.fold.remote(
        sequence=seq,
        num_loops=3,
        num_sampling_steps=50,
        num_diffusion_samples=1,
        seed=0,
        output_format="mmcif",
    )

    n_res = len(resp["plddt"])
    mean_plddt = sum(resp["plddt"]) / n_res if n_res else 0.0
    print(
        f"\nresidues={n_res}  mean_pLDDT={mean_plddt:.2f}  "
        f"pTM={resp['ptm']:.4f}  iptm={resp['iptm']}"
    )
    print(f"structure: {resp['structure_format']}, {len(resp['structure'])} chars")

    out = Path(output_path) if output_path else Path("/tmp/sheaf-esmfold2.cif")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(resp["structure"])
    print(f"wrote {out}")
