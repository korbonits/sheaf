"""Sheaf — unified serving layer for non-text foundation models."""

from __future__ import annotations

# ModalServer and ModelSpec are ray-free; import eagerly.
from sheaf.modal_server import ModalServer
from sheaf.spec import ModelSpec

__version__ = "0.9.0"
__all__ = ["ModalServer", "ModelServer", "ModelSpec"]


def __getattr__(name: str) -> object:
    # Lazy import of ModelServer so that containers without ray installed
    # (e.g. Modal containers with a minimal image) can still use ModalServer.
    if name == "ModelServer":
        from sheaf.server import ModelServer  # noqa: PLC0415

        return ModelServer
    raise AttributeError(f"module 'sheaf' has no attribute {name!r}")
