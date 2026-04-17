"""Batching policies for model-type-aware request scheduling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BatchPolicy(BaseModel):
    """Controls how requests are batched before hitting the model backend.

    max_batch_size: hard cap on requests per batch
    timeout_ms:     max time to wait for a full batch before flushing
    bucket_by:      field name to group on before calling the backend.
                    Requests with the same value of this field are sent to
                    the backend together; requests with different values are
                    dispatched in separate ``batch_predict`` calls within the
                    same Ray Serve batch window.  Useful when sequences of
                    different lengths would otherwise force padding across
                    the whole batch — e.g. ``bucket_by="horizon"`` for time
                    series, ``bucket_by="n_frames"`` for video.

                    ``None`` (default): all requests in a batch window are
                    sent to the backend in a single call.
    """

    max_batch_size: int = Field(default=32, gt=0)
    timeout_ms: int = Field(default=50, gt=0)
    bucket_by: str | None = None


def bucket_requests(
    requests: list[Any],
    bucket_by: str | None,
) -> list[tuple[list[int], list[Any]]]:
    """Group *requests* by the value of field *bucket_by*.

    Returns a list of ``(indices, sub_requests)`` pairs — one per unique
    bucket value, in the order the bucket was first seen.  When *bucket_by*
    is ``None``, returns a single group containing all requests.

    Relative order within each bucket matches the original request list.
    Requests that lack the *bucket_by* attribute (``getattr`` returns
    ``None``) are grouped together under the ``None`` bucket.

    Args:
        requests:  Ordered list of request objects (any type with attrs).
        bucket_by: Name of the field to bucket on, or ``None`` to skip.

    Returns:
        List of ``(original_indices, sub_requests)`` tuples.

    Example::

        reqs = [req(horizon=6), req(horizon=12), req(horizon=6)]
        groups = bucket_requests(reqs, "horizon")
        # groups == [([0, 2], [reqs[0], reqs[2]]),
        #            ([1],    [reqs[1]])]
    """
    if not bucket_by:
        return [(list(range(len(requests))), list(requests))]

    buckets: dict[Any, list[tuple[int, Any]]] = {}
    for i, req in enumerate(requests):
        key = getattr(req, bucket_by, None)
        buckets.setdefault(key, []).append((i, req))

    return [
        ([i for i, _ in group], [r for _, r in group]) for group in buckets.values()
    ]
