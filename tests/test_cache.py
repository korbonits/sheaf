"""Tests for sheaf.cache — CacheConfig, ResponseCache, and server integration."""

from __future__ import annotations

import time as _time
from unittest.mock import patch

import pytest

import tests.stubs  # noqa: F401 — registers _smoke_ts and friends
from sheaf.cache import CacheConfig, ResponseCache

# ---------------------------------------------------------------------------
# CacheConfig
# ---------------------------------------------------------------------------


class TestCacheConfig:
    def test_defaults(self) -> None:
        cfg = CacheConfig()
        assert cfg.enabled is False
        assert cfg.max_size == 1024
        assert cfg.ttl_s is None
        assert cfg.exclude_fields == []

    def test_custom_values(self) -> None:
        cfg = CacheConfig(
            enabled=True, max_size=64, ttl_s=300.0, exclude_fields=["seed"]
        )
        assert cfg.enabled is True
        assert cfg.max_size == 64
        assert cfg.ttl_s == 300.0
        assert cfg.exclude_fields == ["seed"]

    def test_max_size_must_be_gt_zero(self) -> None:
        with pytest.raises(Exception):
            CacheConfig(max_size=0)

    def test_ttl_must_be_gt_zero(self) -> None:
        with pytest.raises(Exception):
            CacheConfig(ttl_s=0.0)


# ---------------------------------------------------------------------------
# ResponseCache.make_key
# ---------------------------------------------------------------------------


class TestMakeKey:
    def _ts_request(
        self,
        history: list | None = None,
        horizon: int = 3,
    ):  # type: ignore[no-untyped-def]
        from sheaf.api.time_series import Frequency, TimeSeriesRequest

        return TimeSeriesRequest(
            model_name="chronos",
            history=history if history is not None else [1.0, 2.0, 3.0],
            horizon=horizon,
            frequency=Frequency.DAILY,
        )

    def test_key_is_64_char_hex(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        key = cache.make_key("dep", self._ts_request())
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_is_deterministic(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        key_a = cache.make_key("dep", self._ts_request())
        key_b = cache.make_key("dep", self._ts_request())
        assert key_a == key_b

    def test_key_differs_by_deployment(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        req = self._ts_request()
        assert cache.make_key("dep-a", req) != cache.make_key("dep-b", req)

    def test_key_excludes_request_id(self) -> None:
        """Two requests with different UUIDs but identical payloads → same key."""
        cache = ResponseCache(CacheConfig(enabled=True))
        r1 = self._ts_request()
        r2 = self._ts_request()
        assert r1.request_id != r2.request_id  # UUIDs differ
        assert cache.make_key("dep", r1) == cache.make_key("dep", r2)

    def test_key_excludes_custom_fields(self) -> None:
        from sheaf.api.time_series import Frequency, TimeSeriesRequest

        cache = ResponseCache(CacheConfig(enabled=True, exclude_fields=["model_name"]))
        r1 = TimeSeriesRequest(
            model_name="model-a",
            history=[1.0, 2.0],
            horizon=3,
            frequency=Frequency.DAILY,
        )
        r2 = TimeSeriesRequest(
            model_name="model-b",
            history=[1.0, 2.0],
            horizon=3,
            frequency=Frequency.DAILY,
        )
        assert cache.make_key("dep", r1) == cache.make_key("dep", r2)

    def test_key_differs_for_different_history(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        r1 = self._ts_request()
        r2 = self._ts_request(history=[9.0, 8.0, 7.0])
        assert cache.make_key("dep", r1) != cache.make_key("dep", r2)

    def test_key_differs_for_different_horizon(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        r1 = self._ts_request()
        r2 = self._ts_request(horizon=12)
        assert cache.make_key("dep", r1) != cache.make_key("dep", r2)


# ---------------------------------------------------------------------------
# ResponseCache get / set
# ---------------------------------------------------------------------------


class TestGetSet:
    def test_get_miss_returns_none(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        assert cache.get("nonexistent") is None

    def test_set_then_get_returns_value(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        cache.set("k1", {"mean": [0.42, 0.42]})
        assert cache.get("k1") == {"mean": [0.42, 0.42]}

    def test_size_tracks_entries(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        assert cache.size == 0
        cache.set("k1", {"v": 1})
        cache.set("k2", {"v": 2})
        assert cache.size == 2

    def test_lru_eviction_at_capacity(self) -> None:
        """When full, the least-recently-used entry is evicted."""
        cache = ResponseCache(CacheConfig(enabled=True, max_size=2))
        cache.set("k1", {"v": 1})
        cache.set("k2", {"v": 2})
        # Touch k1 → k2 becomes LRU.
        cache.get("k1")
        # Inserting k3 should evict k2.
        cache.set("k3", {"v": 3})
        assert cache.size == 2
        assert cache.get("k2") is None
        assert cache.get("k1") == {"v": 1}
        assert cache.get("k3") == {"v": 3}

    def test_update_existing_key_replaces_value(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True))
        cache.set("k1", {"v": 1})
        cache.set("k1", {"v": 99})
        assert cache.get("k1") == {"v": 99}
        assert cache.size == 1

    def test_update_existing_key_promotes_to_mru(self) -> None:
        """Re-setting an existing key should promote it so it is not evicted next."""
        cache = ResponseCache(CacheConfig(enabled=True, max_size=2))
        cache.set("k1", {"v": 1})
        cache.set("k2", {"v": 2})
        # Re-set k1 → k1 is now MRU, k2 is LRU.
        cache.set("k1", {"v": 11})
        cache.set("k3", {"v": 3})  # evicts k2
        assert cache.get("k2") is None
        assert cache.get("k1") == {"v": 11}

    def test_ttl_expiry_returns_none(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True, ttl_s=10.0))
        t0 = _time.monotonic()
        with patch("sheaf.cache.time") as mock_time:
            mock_time.monotonic.return_value = t0
            cache.set("k1", {"v": 1})
            # Advance time beyond TTL.
            mock_time.monotonic.return_value = t0 + 11.0
            assert cache.get("k1") is None
        # Expired entry was removed from the store.
        assert cache.size == 0

    def test_ttl_not_yet_expired_returns_value(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True, ttl_s=10.0))
        t0 = _time.monotonic()
        with patch("sheaf.cache.time") as mock_time:
            mock_time.monotonic.return_value = t0
            cache.set("k1", {"v": 1})
            mock_time.monotonic.return_value = t0 + 5.0  # still within TTL
            assert cache.get("k1") == {"v": 1}

    def test_no_ttl_entry_never_expires(self) -> None:
        cache = ResponseCache(CacheConfig(enabled=True, ttl_s=None))
        cache.set("k1", {"v": 1})
        # No time mocking needed — entries with ttl_s=None never expire.
        assert cache.get("k1") == {"v": 1}


# ---------------------------------------------------------------------------
# Thread safety (smoke test — not a proof)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_set_get(self) -> None:
        import threading

        cache = ResponseCache(CacheConfig(enabled=True, max_size=50))
        errors: list[Exception] = []

        def _worker(i: int) -> None:
            try:
                for j in range(20):
                    key = f"k{i}-{j}"
                    cache.set(key, {"i": i, "j": j})
                    cache.get(key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# SHEAF_CACHE_DISABLED flag
# ---------------------------------------------------------------------------


class TestCacheDisabledFlag:
    def test_module_exports_disabled_flag(self) -> None:
        import sheaf.cache as c

        assert hasattr(c, "_DISABLED")

    def test_disabled_flag_is_bool(self) -> None:
        import sheaf.cache as c

        assert isinstance(c._DISABLED, bool)


# ---------------------------------------------------------------------------
# Integration — _build_asgi_app cache bypass
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    """End-to-end cache tests via _build_asgi_app (no Ray needed)."""

    def _ts_payload(self) -> dict:
        return {
            "model_type": "time_series",
            "model_name": "chronos",
            "history": [1.0, 2.0, 3.0, 4.0],
            "horizon": 3,
            "frequency": "1d",
        }

    def test_cache_hit_bypasses_backend(self) -> None:
        """Second identical request is served from cache; backend not called again."""
        from starlette.testclient import TestClient

        from sheaf.api.base import ModelType
        from sheaf.modal_server import _build_asgi_app
        from sheaf.spec import ModelSpec
        from tests.stubs import SmokeTimeSeriesBackend

        spec = ModelSpec(
            name="ts",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            cache=CacheConfig(enabled=True, max_size=128),
        )

        call_count = 0
        _original_predict = SmokeTimeSeriesBackend.predict

        def _counting_predict(self, request):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return _original_predict(self, request)

        with patch.object(SmokeTimeSeriesBackend, "predict", _counting_predict):
            app = _build_asgi_app([spec])
            client = TestClient(app)

            r1 = client.post("/ts/predict", json=self._ts_payload())
            assert r1.status_code == 200
            r2 = client.post("/ts/predict", json=self._ts_payload())
            assert r2.status_code == 200

        n = call_count
        assert n == 1, f"Backend called {n} times; expected 1 (cache hit)"
        assert r1.json()["mean"] == r2.json()["mean"]

    def test_different_requests_not_cached_together(self) -> None:
        """Requests with different history produce distinct cache entries."""
        from starlette.testclient import TestClient

        from sheaf.api.base import ModelType
        from sheaf.modal_server import _build_asgi_app
        from sheaf.spec import ModelSpec
        from tests.stubs import SmokeTimeSeriesBackend

        spec = ModelSpec(
            name="ts2",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            cache=CacheConfig(enabled=True, max_size=128),
        )

        call_count = 0
        _original_predict = SmokeTimeSeriesBackend.predict

        def _counting_predict(self, request):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return _original_predict(self, request)

        payload_a = self._ts_payload()
        payload_b = {**self._ts_payload(), "history": [9.0, 8.0, 7.0, 6.0]}

        with patch.object(SmokeTimeSeriesBackend, "predict", _counting_predict):
            app = _build_asgi_app([spec])
            client = TestClient(app)
            r1 = client.post("/ts2/predict", json=payload_a)
            r2 = client.post("/ts2/predict", json=payload_b)

        assert r1.status_code == r2.status_code == 200
        n = call_count
        assert n == 2, f"Backend called {n} times; expected 2 (distinct inputs)"

    def test_cache_disabled_by_default(self) -> None:
        """Default CacheConfig (enabled=False) calls backend every time."""
        from starlette.testclient import TestClient

        from sheaf.api.base import ModelType
        from sheaf.modal_server import _build_asgi_app
        from sheaf.spec import ModelSpec
        from tests.stubs import SmokeTimeSeriesBackend

        spec = ModelSpec(
            name="ts3",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            # cache not set → default CacheConfig(enabled=False)
        )

        call_count = 0
        _original_predict = SmokeTimeSeriesBackend.predict

        def _counting_predict(self, request):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return _original_predict(self, request)

        with patch.object(SmokeTimeSeriesBackend, "predict", _counting_predict):
            app = _build_asgi_app([spec])
            client = TestClient(app)
            client.post("/ts3/predict", json=self._ts_payload())
            client.post("/ts3/predict", json=self._ts_payload())

        n = call_count
        assert n == 2, f"Expected 2 backend calls (no cache); got {n}"

    def test_sheaf_cache_disabled_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SHEAF_CACHE_DISABLED=1 skips cache even when spec has enabled=True."""
        import sheaf.cache as c
        import sheaf.modal_server as ms

        monkeypatch.setattr(c, "_DISABLED", True)
        monkeypatch.setattr(ms, "_CACHE_DISABLED", True)

        from starlette.testclient import TestClient

        from sheaf.api.base import ModelType
        from sheaf.modal_server import _build_asgi_app
        from sheaf.spec import ModelSpec
        from tests.stubs import SmokeTimeSeriesBackend

        spec = ModelSpec(
            name="ts4",
            model_type=ModelType.TIME_SERIES,
            backend="_smoke_ts",
            cache=CacheConfig(enabled=True, max_size=128),
        )

        call_count = 0
        _original_predict = SmokeTimeSeriesBackend.predict

        def _counting_predict(self, request):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return _original_predict(self, request)

        with patch.object(SmokeTimeSeriesBackend, "predict", _counting_predict):
            app = _build_asgi_app([spec])
            client = TestClient(app)
            client.post("/ts4/predict", json=self._ts_payload())
            client.post("/ts4/predict", json=self._ts_payload())

        n = call_count
        assert n == 2, f"Expected 2 calls (cache disabled globally); got {n}"
