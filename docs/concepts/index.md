# Concepts

The runtime ideas that hold sheaf-serve together. Each page is the
short version of one design decision.

- **[Batching](batching.md)** — `@serve.batch`, `BatchPolicy`, and
  `bucket_by` for length-variable inputs.
- **[Caching](caching.md)** — opt-in per-deployment LRU response cache
  with optional TTL.
- **[Streaming](streaming.md)** — `POST /{name}/stream` SSE for
  incremental output (FLUX progress, chunked transcripts).
- **[Feast integration](feast.md)** — send a `feature_ref` instead of
  raw history; sheaf-serve resolves features online.
- **[Offline batch jobs](batch.md)** — `BatchRunner`: JSONL in,
  JSONL out, via Ray Data, with stateless-task or warm-actor execution.
- **[Async-job worker](worker.md)** — `SheafWorker`: Redis Streams
  consumer with at-least-once delivery and webhook callbacks.
- **[LoRA multiplexing](lora.md)** — adapter-aware sub-batching for
  diffusion backends; per-request adapter selection on a single
  deployment.
