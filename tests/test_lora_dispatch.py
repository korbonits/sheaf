"""Integration tests for LoRA dispatch through _batch_predict logic.

Reproduces the bucket-and-dispatch logic that ``_SheafDeployment._batch_predict``
runs when ``ModelSpec.lora`` is set, without requiring Ray Serve.  Verifies:

  - ``set_active_adapters`` is called once per resolved bucket
  - results are reassembled in original arrival order
  - requests with empty adapters and a default share a bucket with explicit
    requests for the same default
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sheaf.api.diffusion import DiffusionRequest, DiffusionResponse
from sheaf.lora import (
    LoRAAdapter,
    LoRAConfig,
    bucket_with_adapter_resolution,
)


def _req(adapters: list[str] | None = None, weights: list[float] | None = None):
    return DiffusionRequest(
        model_name="flux-schnell",
        prompt="a cat",
        adapters=adapters or [],
        adapter_weights=weights,
    )


def _resp(req: DiffusionRequest) -> DiffusionResponse:
    return DiffusionResponse(
        request_id=req.request_id,
        model_name=req.model_name,
        image_b64="x",
        height=req.height,
        width=req.width,
        seed=42,
    )


def _backend() -> MagicMock:
    b = MagicMock()
    b.set_active_adapters = MagicMock()

    async def _abp(reqs: list[Any]) -> list[Any]:
        return [_resp(r) for r in reqs]

    b.async_batch_predict = AsyncMock(side_effect=_abp)
    return b


async def _dispatch(
    backend: MagicMock, requests: list[DiffusionRequest], lora: LoRAConfig
) -> list[dict[str, Any]]:
    """Reimplement the LoRA-aware path of _batch_predict without Ray."""
    groups = bucket_with_adapter_resolution(requests, lora)
    if len(groups) == 1 and not groups[0][2]:
        responses = await backend.async_batch_predict(groups[0][1])
        return [r.model_dump(mode="json") for r in responses]
    slot: dict[int, dict[str, Any]] = {}
    for indices, sub_reqs, names, weights in groups:
        backend.set_active_adapters(names, weights)
        bucket_responses = await backend.async_batch_predict(sub_reqs)
        for idx, resp in zip(indices, bucket_responses):
            slot[idx] = resp.model_dump(mode="json")
    return [slot[i] for i in range(len(requests))]


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.get_event_loop().run_until_complete(coro)


_LORA = LoRAConfig(
    adapters={
        "sketch": LoRAAdapter(source="/tmp/sketch", weight=0.8),
        "watercolor": LoRAAdapter(source="/tmp/wc", weight=1.0),
    },
    default="sketch",
)


class TestLoRADispatch:
    def test_homogeneous_explicit_one_set_call(self) -> None:
        reqs = [_req(["sketch"]), _req(["sketch"])]
        backend = _backend()
        results = _run(_dispatch(backend, reqs, _LORA))
        assert backend.set_active_adapters.call_count == 1
        backend.set_active_adapters.assert_called_with(["sketch"], [0.8])
        assert backend.async_batch_predict.call_count == 1
        assert len(results) == 2

    def test_default_and_explicit_share_bucket(self) -> None:
        # Default is "sketch"; both requests resolve to ["sketch"], [0.8]
        # → one set_active_adapters call, one async_batch_predict call.
        reqs = [_req(), _req(["sketch"])]
        backend = _backend()
        _run(_dispatch(backend, reqs, _LORA))
        assert backend.set_active_adapters.call_count == 1
        assert backend.async_batch_predict.call_count == 1

    def test_two_adapters_two_set_calls(self) -> None:
        reqs = [
            _req(["sketch"]),
            _req(["watercolor"]),
            _req(["sketch"]),
        ]
        backend = _backend()
        results = _run(_dispatch(backend, reqs, _LORA))
        assert backend.set_active_adapters.call_count == 2
        # Order preserved in the output.
        assert len(results) == 3
        # The two "sketch" requests went to one batch.
        seen_calls = backend.async_batch_predict.call_args_list
        sizes = sorted(len(c.args[0]) for c in seen_calls)
        assert sizes == [1, 2]  # one bucket of 2 (sketch), one of 1 (watercolor)

    def test_results_in_original_order(self) -> None:
        reqs = [
            _req(["sketch"]),  # 0
            _req(["watercolor"]),  # 1
            _req(["sketch"]),  # 2
        ]
        backend = _backend()
        results = _run(_dispatch(backend, reqs, _LORA))
        # request_id round-trips, so we can verify ordering.
        assert results[0]["request_id"] == str(reqs[0].request_id)
        assert results[1]["request_id"] == str(reqs[1].request_id)
        assert results[2]["request_id"] == str(reqs[2].request_id)

    def test_no_adapter_no_default_skips_set(self) -> None:
        cfg = LoRAConfig(
            adapters={"sketch": LoRAAdapter(source="/tmp/sketch")}, default=None
        )
        reqs = [_req(), _req()]
        backend = _backend()
        _run(_dispatch(backend, reqs, cfg))
        # No adapters resolved → fast path: no set_active_adapters call.
        assert backend.set_active_adapters.call_count == 0
        assert backend.async_batch_predict.call_count == 1

    def test_request_weight_overrides_default(self) -> None:
        reqs = [_req(["sketch"], [0.3])]
        backend = _backend()
        _run(_dispatch(backend, reqs, _LORA))
        backend.set_active_adapters.assert_called_with(["sketch"], [0.3])

    def test_unknown_adapter_raises(self) -> None:
        reqs = [_req(["bogus"])]
        backend = _backend()
        with pytest.raises(ValueError, match="Unknown adapter"):
            _run(_dispatch(backend, reqs, _LORA))
