"""Ray Serve Application factory for KubeRay deployment.

KubeRay's ``RayService`` resource references a Ray Serve Application via
``serveConfigV2.applications[].import_path: "app:app"`` — meaning Ray
imports this module and reads the ``app`` attribute as the deployable
application.

``sheaf.build_app(spec)`` does the heavy lifting: it returns the same
Ray Serve Application that ``ModelServer.run()`` deploys internally,
constructed from a ``ModelSpec``.  Edit the spec below to match your
deployment, then bake this file into your container image alongside
the official sheaf-serve base image.
"""

from __future__ import annotations

from sheaf import build_app
from sheaf.api.base import ModelType
from sheaf.spec import ModelSpec, ResourceConfig

spec = ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
    },
    resources=ResourceConfig(num_cpus=1),
)

app = build_app(spec)
