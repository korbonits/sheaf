"""Tests for resolve_active_adapters + bucket_with_adapter_resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sheaf.lora import (
    LoRAAdapter,
    LoRAConfig,
    bucket_with_adapter_resolution,
    resolve_active_adapters,
)


@dataclass
class _Req:
    adapters: list[str] = field(default_factory=list)
    adapter_weights: list[float] | None = None


def _cfg(default: str | None = None) -> LoRAConfig:
    return LoRAConfig(
        adapters={
            "sketch": LoRAAdapter(source="/loras/sketch", weight=0.8),
            "watercolor": LoRAAdapter(source="/loras/wc", weight=1.0),
        },
        default=default,
    )


class TestResolveActiveAdapters:
    def test_no_lora_config_returns_empty(self) -> None:
        names, weights = resolve_active_adapters(_Req(adapters=["x"]), None)
        assert names == []
        assert weights == []

    def test_request_adapters_uses_spec_default_weight(self) -> None:
        names, weights = resolve_active_adapters(_Req(adapters=["sketch"]), _cfg())
        assert names == ["sketch"]
        assert weights == [0.8]  # from LoRAAdapter.weight

    def test_request_adapter_weights_override_spec(self) -> None:
        names, weights = resolve_active_adapters(
            _Req(adapters=["sketch"], adapter_weights=[0.3]), _cfg()
        )
        assert names == ["sketch"]
        assert weights == [0.3]

    def test_default_used_when_request_empty(self) -> None:
        names, weights = resolve_active_adapters(_Req(), _cfg(default="sketch"))
        assert names == ["sketch"]
        assert weights == [0.8]

    def test_no_default_no_request_adapters_returns_empty(self) -> None:
        names, weights = resolve_active_adapters(_Req(), _cfg(default=None))
        assert names == []
        assert weights == []

    def test_unknown_adapter_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter"):
            resolve_active_adapters(_Req(adapters=["nope"]), _cfg())

    def test_fusion_uses_per_adapter_weights(self) -> None:
        names, weights = resolve_active_adapters(
            _Req(adapters=["sketch", "watercolor"]), _cfg()
        )
        assert names == ["sketch", "watercolor"]
        assert weights == [0.8, 1.0]

    def test_request_omits_unknown_lora_when_none(self) -> None:
        # lora is None and request has adapters set — by spec the helper
        # returns empty; validation that this is an error is the caller's job.
        names, weights = resolve_active_adapters(_Req(adapters=["sketch"]), None)
        assert (names, weights) == ([], [])


class TestBucketWithAdapterResolution:
    def test_no_lora_single_bucket(self) -> None:
        reqs = [_Req(), _Req(adapters=["x"])]  # adapters ignored when lora=None
        groups = bucket_with_adapter_resolution(reqs, None)
        assert len(groups) == 1
        indices, sub_reqs, names, weights = groups[0]
        assert indices == [0, 1]
        assert names == []
        assert weights == []

    def test_resolves_then_buckets(self) -> None:
        """Request with empty adapters + spec default and request explicit
        same name/weight should land in the SAME bucket after resolution."""
        cfg = _cfg(default="sketch")
        reqs = [
            _Req(),  # → ["sketch"], [0.8]
            _Req(adapters=["sketch"]),  # → ["sketch"], [0.8]
            _Req(adapters=["sketch"], adapter_weights=[0.8]),  # → same key
        ]
        groups = bucket_with_adapter_resolution(reqs, cfg)
        assert len(groups) == 1
        indices, _, names, weights = groups[0]
        assert sorted(indices) == [0, 1, 2]
        assert names == ["sketch"]
        assert weights == [0.8]

    def test_different_adapters_separate_buckets(self) -> None:
        cfg = _cfg(default="sketch")
        reqs = [
            _Req(adapters=["sketch"]),
            _Req(adapters=["watercolor"]),
            _Req(),  # default sketch
        ]
        groups = bucket_with_adapter_resolution(reqs, cfg)
        assert len(groups) == 2
        by_name = {g[2][0]: g for g in groups}
        assert sorted(by_name["sketch"][0]) == [0, 2]
        assert by_name["watercolor"][0] == [1]

    def test_same_adapter_different_weights_separate(self) -> None:
        cfg = _cfg()
        reqs = [
            _Req(adapters=["sketch"], adapter_weights=[0.5]),
            _Req(adapters=["sketch"], adapter_weights=[1.0]),
        ]
        groups = bucket_with_adapter_resolution(reqs, cfg)
        assert len(groups) == 2

    def test_fusion_set_buckets_separately_from_solo(self) -> None:
        cfg = _cfg()
        reqs = [
            _Req(adapters=["sketch", "watercolor"]),
            _Req(adapters=["sketch"]),
        ]
        groups = bucket_with_adapter_resolution(reqs, cfg)
        assert len(groups) == 2

    def test_unknown_adapter_propagates(self) -> None:
        with pytest.raises(ValueError, match="Unknown adapter"):
            bucket_with_adapter_resolution([_Req(adapters=["bogus"])], _cfg())
