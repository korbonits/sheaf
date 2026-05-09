# Streaming (SSE)

Each deployment exposes `POST /{name}/stream` for incremental output:

- FLUX progress events (per-step latents → final image)
- Chunked transcription (audio segments as they decode)
- Token-level generation, partial samples — anything where the client
  benefits from output before the whole inference completes

The wire format is plain [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events):
each event is a `data: {json}\n\n` line. Clients strip the `data:`
prefix and JSON-decode the body.

## Default behaviour

Every backend implements `stream_predict` as an async generator. The
default base implementation yields a single `{"type": "result",
"done": true, ...response}` event after `async_predict()` completes.

Backends that benefit from chunking override the method. Example
shape (from `FluxBackend`):

```json
{"type": "progress", "step": 5, "total_steps": 50}
{"type": "progress", "step": 10, "total_steps": 50}
...
{"type": "result", "image_b64": "...", "done": true}
```

## Calling it from the client

```python
from sheaf.client import AsyncSheafClient
from sheaf.api.diffusion import DiffusionRequest

async with AsyncSheafClient(base_url="http://127.0.0.1:8000") as client:
    async for event in client.stream(
        "flux",
        DiffusionRequest(
            model_name="flux",
            prompt="a sheaf, mathematical concept, line drawing",
            num_inference_steps=20,
        ),
    ):
        if event["type"] == "progress":
            print(f"step {event['step']}/{event['total_steps']}")
        elif event["type"] == "result":
            save_image(event["image_b64"])
            break
```

`AsyncSheafClient.stream` is the only async-only method on the
client — sync streaming is available via raw httpx, but the typed
client is async-first by design here.

## What streaming bypasses

- **No batching.** `@serve.batch` produces a single response; SSE
  streams are per-request by nature, so the streaming endpoint calls
  `backend.stream_predict()` directly.
- **No cache.** The stream is ephemeral; caching it would either
  require buffering the full output (defeats the purpose) or replaying
  events with synthetic timing (lies about behaviour).
- **Feast still resolves.** The same `feature_ref → resolved history`
  lookup runs before the backend call.

## Errors mid-stream

If the backend raises after the first event has been sent, the HTTP
response is already committed — sheaf-serve emits a final
`{"type": "error", "error": "..."}` event instead of crashing the SSE
stream. Clients should check `event["type"]` on every received frame.

## LoRA and concurrent streams

`pipeline.set_adapters` on diffusers is process-global state. The
streaming endpoint applies the adapter selection per request before
`stream_predict()`, but two concurrent streams against the same
deployment with different adapters race the global state. Since
streaming is per-request anyway and benefits little from concurrency,
the operational rule is **one stream at a time per replica** when
LoRA is in use.
