"""Tests for bucket_by batching — bucket_requests helper and dispatch logic."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sheaf.scheduling.batch import BatchPolicy, bucket_requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts_req(horizon: int):  # type: ignore[no-untyped-def]
    from sheaf.api.time_series import Frequency, TimeSeriesRequest

    return TimeSeriesRequest(
        model_name="m",
        history=[1.0, 2.0],
        horizon=horizon,
        frequency=Frequency.DAILY,
    )


def _ts_resp(req):  # type: ignore[no-untyped-def]
    from sheaf.api.time_series import TimeSeriesResponse

    return TimeSeriesResponse(
        request_id=req.request_id,
        model_name=req.model_name,
        horizon=req.horizon,
        frequency=req.frequency.value,
        mean=[0.0] * req.horizon,
    )


# ---------------------------------------------------------------------------
# BatchPolicy
# ---------------------------------------------------------------------------


class TestBatchPolicy:
    def test_defaults(self) -> None:
        bp = BatchPolicy()
        assert bp.max_batch_size == 32
        assert bp.timeout_ms == 50
        assert bp.bucket_by is None

    def test_bucket_by_field(self) -> None:
        bp = BatchPolicy(bucket_by="horizon")
        assert bp.bucket_by == "horizon"


# ---------------------------------------------------------------------------
# bucket_requests — no bucketing
# ---------------------------------------------------------------------------


class TestBucketRequestsNoBucket:
    def test_none_returns_single_group_with_all(self) -> None:
        reqs = [_ts_req(3), _ts_req(6), _ts_req(12)]
        groups = bucket_requests(reqs, None)
        assert len(groups) == 1
        assert groups[0][0] == [0, 1, 2]
        assert groups[0][1] == reqs

    def test_empty_string_treated_as_falsy_single_group(self) -> None:
        reqs = [_ts_req(3), _ts_req(6)]
        groups = bucket_requests(reqs, "")
        assert len(groups) == 1
        assert groups[0][0] == [0, 1]

    def test_empty_request_list_no_bucket(self) -> None:
        groups = bucket_requests([], None)
        assert groups == [([], [])]

    def test_single_request_no_bucket(self) -> None:
        req = _ts_req(6)
        groups = bucket_requests([req], None)
        assert len(groups) == 1
        assert groups[0][0] == [0]
        assert groups[0][1] == [req]


# ---------------------------------------------------------------------------
# bucket_requests — with bucketing
# ---------------------------------------------------------------------------


class TestBucketRequestsWithBucket:
    def test_all_same_value_single_group(self) -> None:
        reqs = [_ts_req(6), _ts_req(6), _ts_req(6)]
        groups = bucket_requests(reqs, "horizon")
        assert len(groups) == 1
        indices, sub = groups[0]
        assert indices == [0, 1, 2]
        assert sub == reqs

    def test_all_different_values_n_groups(self) -> None:
        reqs = [_ts_req(3), _ts_req(6), _ts_req(12)]
        groups = bucket_requests(reqs, "horizon")
        assert len(groups) == 3
        all_indices = sorted(i for indices, _ in groups for i in indices)
        assert all_indices == [0, 1, 2]

    def test_mixed_buckets_correct_grouping(self) -> None:
        reqs = [
            _ts_req(6),  # 0 — bucket 6
            _ts_req(12),  # 1 — bucket 12
            _ts_req(6),  # 2 — bucket 6
            _ts_req(12),  # 3 — bucket 12
            _ts_req(3),  # 4 — bucket 3
        ]
        groups = bucket_requests(reqs, "horizon")
        assert len(groups) == 3
        by_bucket = {sub[0].horizon: (indices, sub) for indices, sub in groups}
        assert by_bucket[6][0] == [0, 2]
        assert by_bucket[12][0] == [1, 3]
        assert by_bucket[3][0] == [4]

    def test_order_preserved_within_bucket(self) -> None:
        reqs = [_ts_req(6), _ts_req(12), _ts_req(6)]
        groups = bucket_requests(reqs, "horizon")
        by_bucket = {sub[0].horizon: (indices, sub) for indices, sub in groups}
        # Within bucket 6: original positions 0 and 2, in that order.
        assert by_bucket[6][0] == [0, 2]
        assert by_bucket[6][1] == [reqs[0], reqs[2]]

    def test_first_seen_bucket_ordering(self) -> None:
        """Buckets appear in the order their first request arrived."""
        reqs = [_ts_req(12), _ts_req(6), _ts_req(3)]
        groups = bucket_requests(reqs, "horizon")
        bucket_values = [sub[0].horizon for _, sub in groups]
        assert bucket_values == [12, 6, 3]

    def test_missing_field_falls_into_none_bucket(self) -> None:
        """Requests lacking the bucket field are grouped under the None key."""

        class Bare:
            pass  # no horizon attribute

        r1, r2 = Bare(), Bare()
        groups = bucket_requests([r1, r2], "horizon")
        assert len(groups) == 1
        assert groups[0][0] == [0, 1]

    def test_single_request_with_bucket(self) -> None:
        req = _ts_req(6)
        groups = bucket_requests([req], "horizon")
        assert len(groups) == 1
        assert groups[0][0] == [0]
        assert groups[0][1] == [req]

    def test_indices_cover_all_positions(self) -> None:
        """Every original index appears exactly once across all buckets."""
        reqs = [_ts_req(h) for h in [3, 6, 12, 6, 3, 12, 6]]
        groups = bucket_requests(reqs, "horizon")
        all_indices = sorted(i for indices, _ in groups for i in indices)
        assert all_indices == list(range(len(reqs)))


# ---------------------------------------------------------------------------
# Dispatch integration — mock backend, verify call count and result order
# ---------------------------------------------------------------------------


class TestBucketDispatch:
    """Verify that _batch_predict dispatches one call per bucket and
    reassembles results in the correct order.  Uses asyncio directly to
    drive the coroutine without needing Ray Serve."""

    def _make_spec(self, bucket_by: str | None) -> MagicMock:
        spec = MagicMock()
        spec.name = "test-dep"
        spec.batch_policy.bucket_by = bucket_by
        return spec

    def _make_backend(self, requests: list[Any]) -> MagicMock:
        """Backend whose async_batch_predict echoes _ts_resp for each req."""
        backend = MagicMock()

        async def _abp(reqs: list[Any]) -> list[Any]:
            return [_ts_resp(r) for r in reqs]

        backend.async_batch_predict = AsyncMock(side_effect=_abp)
        return backend

    def _run(self, coro):  # type: ignore[no-untyped-def]
        return asyncio.get_event_loop().run_until_complete(coro)

    async def _dispatch(
        self, requests: list[Any], bucket_by: str | None
    ) -> list[dict[str, Any]]:
        """Reproduce the _batch_predict logic without Ray decorators."""
        backend = self._make_backend(requests)
        groups = bucket_requests(requests, bucket_by)
        if len(groups) == 1:
            responses = await backend.async_batch_predict(groups[0][1])
            return [r.model_dump(mode="json") for r in responses]

        slot: dict[int, dict[str, Any]] = {}
        for indices, sub_reqs in groups:
            bucket_responses = await backend.async_batch_predict(sub_reqs)
            for idx, resp in zip(indices, bucket_responses):
                slot[idx] = resp.model_dump(mode="json")
        return [slot[i] for i in range(len(requests))]

    @pytest.mark.asyncio
    async def test_no_bucket_single_backend_call(self) -> None:
        reqs = [_ts_req(6), _ts_req(12), _ts_req(3)]
        backend = self._make_backend(reqs)
        groups = bucket_requests(reqs, None)
        responses = await backend.async_batch_predict(groups[0][1])
        results = [r.model_dump(mode="json") for r in responses]

        backend.async_batch_predict.assert_called_once()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_bucket_by_horizon_two_calls(self) -> None:
        """Three requests: horizon 6, 12, 6 → two backend calls."""
        reqs = [_ts_req(6), _ts_req(12), _ts_req(6)]
        backend = self._make_backend(reqs)
        groups = bucket_requests(reqs, "horizon")

        results: list[dict[str, Any] | None] = [None] * len(reqs)
        for indices, sub_reqs in groups:
            bucket_responses = await backend.async_batch_predict(sub_reqs)
            for idx, resp in zip(indices, bucket_responses):
                results[idx] = resp.model_dump(mode="json")

        assert backend.async_batch_predict.call_count == 2
        # Each call got requests of the same horizon.
        call_args = backend.async_batch_predict.call_args_list
        horizons_per_call = [{r.horizon for r in call.args[0]} for call in call_args]
        assert all(len(h) == 1 for h in horizons_per_call)

    @pytest.mark.asyncio
    async def test_results_in_original_order(self) -> None:
        """Results must align with the original request list, not bucket order."""
        reqs = [
            _ts_req(6),  # 0
            _ts_req(12),  # 1
            _ts_req(6),  # 2
        ]
        results = await self._dispatch(reqs, "horizon")
        assert len(results) == 3
        assert results[0]["horizon"] == 6
        assert results[1]["horizon"] == 12
        assert results[2]["horizon"] == 6

    @pytest.mark.asyncio
    async def test_five_mixed_requests_correct_order(self) -> None:
        reqs = [
            _ts_req(3),  # 0
            _ts_req(12),  # 1
            _ts_req(3),  # 2
            _ts_req(6),  # 3
            _ts_req(12),  # 4
        ]
        results = await self._dispatch(reqs, "horizon")
        expected_horizons = [3, 12, 3, 6, 12]
        actual_horizons = [r["horizon"] for r in results]
        assert actual_horizons == expected_horizons
