# Caching

An opt-in, in-process LRU cache attached per deployment. Disabled by
default — turn it on per `ModelSpec` if your workload has repeated
inputs (forecasting the same series multiple times in an hour, the
same prompt with the same seed, etc.).

```python
from sheaf import ModelSpec
from sheaf.api.base import ModelType
from sheaf.cache import CacheConfig

spec = ModelSpec(
    name="forecaster",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    cache=CacheConfig(
        enabled=True,
        max_size=1024,    # entries; LRU evicts beyond this
        ttl_seconds=300,  # optional; entries expire after 5 min
    ),
)
```

The cache key is `SHA-256(deployment_name || JSON-canonical request)`,
with `request_id` always excluded so two calls that differ only in
client-generated UUID share a hit.

You can exclude additional fields via `CacheConfig.exclude_fields`.
A common case is diffusion `seed`: same prompt + different seed should
miss, but same prompt + same seed should hit on a retry — leave `seed`
in the key.

## Where the cache sits in the request path

```
HTTP request
   │
   ├─▶  Feast resolution        (if feature_ref)
   │
   ├─▶  Cache lookup            (key includes resolved features)
   │       │
   │       └── HIT → return decoded response
   │
   ├─▶  Batch dispatch          (@serve.batch + bucket_by)
   │
   ├─▶  backend.batch_predict
   │
   └─▶  Cache store + HTTP response
```

The lookup happens **after** Feast resolves features, so the cache key
reflects actual values, not a feature reference. Two requests for the
same entity at different times (with different resolved features)
produce distinct entries, as they should.

## Process-wide disable

Set `SHEAF_CACHE_DISABLED=1` to skip every cache regardless of spec
config. Useful in integration tests where you want every request to
exercise the backend.

## Reference

Full schema in the [Caching API reference](../api-reference/cache.md).
