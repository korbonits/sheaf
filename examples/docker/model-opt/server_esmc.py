"""Serve ESMC 6B under the ESM C kit's exact mode with Ray Serve.

Runs inside the image built by build_esmc.sh.  ``SHEAF_ESMC_MODE`` selects
off / exact (default exact); the compile cache lives under
``SHEAF_MODEL_OPT_JIT_ROOT`` (mount a node-local, private directory there).
A kit refusal fails the deployment at startup with the kit's NOT ACTIVE line.
"""

from __future__ import annotations

import os
import threading

from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType
from sheaf.model_opt import ModelOptConfig
from sheaf.scheduling.batch import BatchPolicy
from sheaf.spec import ResourceConfig

spec = ModelSpec(
    name="esmc-6b",
    model_type=ModelType.PROTEIN_LANGUAGE,
    backend="esmc",
    backend_kwargs={"model_name": "esmc_6b", "device": "cuda"},
    resources=ResourceConfig(num_gpus=1),
    batch_policy=BatchPolicy(max_batch_size=8, timeout_ms=20),
    model_opt=ModelOptConfig(
        mode=os.environ.get("SHEAF_ESMC_MODE", "exact"),  # type: ignore[arg-type]
        jit_root=os.environ.get("SHEAF_MODEL_OPT_JIT_ROOT"),
    ),
)

if __name__ == "__main__":
    ModelServer(models=[spec], host="0.0.0.0", port=8000).run()
    # run() returns after deploying; the Serve replicas live as long as this
    # driver does.
    threading.Event().wait()
