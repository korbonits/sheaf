"""Sheaf — unified serving layer for non-text foundation models."""

from sheaf.server import ModelServer
from sheaf.spec import ModelSpec

__version__ = "0.1.0"
__all__ = ["ModelServer", "ModelSpec"]
