"""Backend registry — separated to avoid circular imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sheaf.backends.base import ModelBackend

_BACKEND_REGISTRY: dict[str, type["ModelBackend"]] = {}


def register_backend(name: str):
    """Decorator to register a ModelBackend implementation by name.

    Example:
        @register_backend("chronos2")
        class Chronos2Backend(ModelBackend):
            ...
    """
    def decorator(cls: type["ModelBackend"]) -> type["ModelBackend"]:
        _BACKEND_REGISTRY[name] = cls
        return cls
    return decorator
