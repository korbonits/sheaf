# Feast integration

[Feast](https://feast.dev) is a feature store: it tracks feature
definitions, materialises them into an online serving layer (Redis,
DynamoDB, etc.), and lets clients ask "give me the freshest features
for entity X." Sheaf-serve treats Feast as a first-class input
primitive — clients can send a `feature_ref` instead of raw history,
and the deployment resolves it inline before inference.

```bash
pip install "sheaf-serve[feast,time-series]"
```

```python
from sheaf import ModelServer, ModelSpec
from sheaf.api.base import ModelType

server = ModelServer(models=[
    ModelSpec(
        name="forecaster",
        model_type=ModelType.TIME_SERIES,
        backend="chronos2",
        feast_repo_path="/path/to/feature_repo",  # contains feature_store.yaml
    ),
])
server.run()
```

The deployment now accepts requests of two shapes:

**Raw history (works without Feast):**

```json
{
  "model_type": "time_series",
  "model_name": "forecaster",
  "history": [312, 298, 275, ...],
  "horizon": 12,
  "frequency": "1h"
}
```

**Feature reference (resolved server-side):**

```json
{
  "model_type": "time_series",
  "model_name": "forecaster",
  "feature_ref": {
    "feature_view": "hourly_demand_v1",
    "feature_name": "kwh_history",
    "entity_key": {"region": "us-west-2"}
  },
  "horizon": 12,
  "frequency": "1h"
}
```

The two are mutually exclusive — a request with both is rejected by
the `TimeSeriesRequest` validator.

## What happens server-side

```
HTTP request with feature_ref
   │
   ├─▶  FeastResolver.resolve(feature_ref)
   │       │
   │       └── feature_store.get_online_features([entity_key]) → list[float]
   │
   ├─▶  Build a TimeSeriesRequest with the resolved history
   │
   └─▶  ... continues through cache → batch → backend
```

The resolver wraps `feast.FeatureStore`. It is constructed once per
deployment, at `_SheafDeployment.__init__`, with the `feast_repo_path`
from the spec. Errors at resolution time (entity not found, store
unavailable) become 422s on the request — see the
[FeastResolver reference](../api-reference/api.md) for the exact
exception types.

## Why server-side resolution

Three reasons:

1. **Cache key includes resolved values.** A request that hits the
   cache after Feast resolves the same features for the same entity
   gets a real hit — not a miss because the wire payload differed.
2. **One round-trip from the client.** Clients don't need to know
   about Feast at all; they send an entity key, get a forecast back.
3. **Centralised feature definitions.** Feature views live with the
   model spec, not scattered across every caller's request-construction
   code.

## Reference

The Feast wiring lives in `sheaf.integrations.feast`. The request-side
field is `TimeSeriesRequest.feature_ref` — schema in the
[time series API reference](../api-reference/api.md#sheaf.api.time_series).
