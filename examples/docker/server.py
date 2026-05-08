"""Minimal ModelServer entrypoint for the example Docker image.

Serves Chronos-Bolt-Tiny on CPU — small enough to run anywhere, real
enough to validate the full Sheaf serving stack inside a container.

Replace the spec list with your own deployment(s); the rest of the file
is boilerplate.
"""

from __future__ import annotations

from sheaf import ModelServer
from sheaf.api.base import ModelType
from sheaf.spec import ModelSpec, ResourceConfig

server = ModelServer(
    models=[
        ModelSpec(
            name="chronos",
            model_type=ModelType.TIME_SERIES,
            backend="chronos2",
            backend_kwargs={
                "model_id": "amazon/chronos-bolt-tiny",
                "device_map": "cpu",
            },
            resources=ResourceConfig(num_cpus=1),
        ),
    ],
    host="0.0.0.0",  # bind to all interfaces so the published port reaches us
    port=8000,
)


if __name__ == "__main__":
    server.run()
    # ModelServer.run() returns immediately after deploying to Ray Serve.
    # Ray Serve actors persist as long as this driver process does, so we
    # block forever.  threading.Event().wait() is cross-platform and
    # interrupts cleanly on SIGTERM (e.g. `docker stop`) and SIGINT.
    import threading

    threading.Event().wait()
