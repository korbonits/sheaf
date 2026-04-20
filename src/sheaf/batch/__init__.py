"""Offline batch inference — JSONL → JSONL via Ray Data.

Public surface::

    from sheaf.batch import BatchRunner, BatchSpec, JsonlSource, JsonlSink

Install with: pip install 'sheaf-serve[batch]'
"""

from sheaf.batch.runner import BatchRunner
from sheaf.batch.spec import BatchSpec, JsonlSink, JsonlSource

__all__ = ["BatchRunner", "BatchSpec", "JsonlSink", "JsonlSource"]
