"""Side-effect imports that populate the backend registry.

Called from both ``sheaf.server._SheafDeployment.__init__`` and
``sheaf.batch.runner._build_backend`` so Ray worker processes reliably
see the full built-in registry regardless of how the driver was started.

Each ``import sheaf.backends.<name>`` runs its ``@register_backend``
decorator.  Heavy optional deps remain lazy inside each backend's
``load()``.
"""

from __future__ import annotations

import importlib
import os


def register_builtin_backends() -> None:
    """Import every built-in backend module.  Idempotent."""
    import sheaf.backends.bark  # noqa: F401
    import sheaf.backends.chronos  # noqa: F401
    import sheaf.backends.depth_anything  # noqa: F401
    import sheaf.backends.detr  # noqa: F401
    import sheaf.backends.dinov2  # noqa: F401
    import sheaf.backends.esm3  # noqa: F401
    import sheaf.backends.faster_whisper  # noqa: F401
    import sheaf.backends.flux  # noqa: F401
    import sheaf.backends.graphcast  # noqa: F401
    import sheaf.backends.imagebind  # noqa: F401
    import sheaf.backends.kokoro  # noqa: F401
    import sheaf.backends.mace  # noqa: F401
    import sheaf.backends.moirai  # noqa: F401
    import sheaf.backends.molformer  # noqa: F401
    import sheaf.backends.musicgen  # noqa: F401
    import sheaf.backends.nucleotide_transformer  # noqa: F401
    import sheaf.backends.open_clip  # noqa: F401
    import sheaf.backends.pointnet  # noqa: F401
    import sheaf.backends.prithvi  # noqa: F401
    import sheaf.backends.raft  # noqa: F401
    import sheaf.backends.sam2  # noqa: F401
    import sheaf.backends.sdxl  # noqa: F401
    import sheaf.backends.tabpfn  # noqa: F401
    import sheaf.backends.timesfm  # noqa: F401
    import sheaf.backends.videomae  # noqa: F401
    import sheaf.backends.vitpose  # noqa: F401
    import sheaf.backends.whisper  # noqa: F401


def register_extra_backends() -> None:
    """Import any modules listed in ``SHEAF_EXTRA_BACKENDS`` (comma-separated)."""
    for _mod in os.environ.get("SHEAF_EXTRA_BACKENDS", "").split(","):
        if _mod.strip():
            importlib.import_module(_mod.strip())
