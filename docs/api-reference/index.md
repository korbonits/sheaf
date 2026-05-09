# API reference

Auto-generated from the live Pydantic models and class docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). Source of truth is the
code in `src/sheaf/`; this section never drifts from a release because
the docs build imports the package itself.

| Module | What's there |
|---|---|
| **[Specs](spec.md)** | `ModelSpec`, `ResourceConfig` |
| **[API contracts](api.md)** | Per-model-type request / response schemas, `AnyRequest`, `AnyResponse`, `BaseRequest`, `BaseResponse`, `ModelType` |
| **[Scheduling](scheduling.md)** | `BatchPolicy`, `bucket_requests` |
| **[Caching](cache.md)** | `CacheConfig`, `ResponseCache` |
| **[LoRA](lora.md)** | `LoRAAdapter`, `LoRAConfig`, `parse_source`, `resolve_active_adapters` |
| **[Batch runner](batch.md)** | `BatchSpec`, `BatchRunner`, `JsonlSource`, `JsonlSink` |
| **[Worker](worker.md)** | `WorkerSpec`, `SheafWorker`, `JobQueueClient`, `JobQueue`, `ResultStore`, `RedisStreamsQueue`, `RedisHashResultStore` |
| **[Client](client.md)** | `SheafClient`, `AsyncSheafClient`, `RetryConfig`, `SheafError` |

## What's not auto-doc'd

Backends in `sheaf.backends.*` aren't auto-rendered here — the public
contract is the typed request / response shape, not the backend's
internals. The [Models](../models/index.md) catalogue lists every
backend's `backend=` name, extras flag, and pointer to the runnable
`examples/quickstart_*.py`. That's the right entry point for users.

For backend implementers (writing your own), see
[CONTRIBUTING.md](https://github.com/korbonits/sheaf/blob/main/CONTRIBUTING.md)
— the abstract base lives at `sheaf.backends.base.ModelBackend`.
