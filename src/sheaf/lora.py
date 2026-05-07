"""LoRA adapter configuration for Sheaf deployments.

Each ``ModelSpec`` may declare a set of named LoRA adapters via
``ModelSpec.lora = LoRAConfig(...)``.  Adapters are loaded once at deploy time
and selected per-request via the ``adapters`` / ``adapter_weights`` fields on
the request.

Adapter sources
---------------
``LoRAAdapter.source`` accepts two forms:

- A local filesystem path: ``/models/loras/sketch.safetensors`` or a directory
  containing the adapter weights file.
- A HuggingFace Hub reference: ``hf:org/repo`` or
  ``hf:org/repo:weight_file.safetensors`` when the repo contains multiple
  adapter weight files.

Backends parse ``hf:`` themselves; the spec only validates that the source
string is non-empty.

Bucketing
---------
``pipeline.set_adapters(...)`` (Diffusers / PEFT) is process-global state on
the pipeline.  Two concurrent requests inside the same Ray Serve batch window
that select different adapters would race.  ``BatchPolicy.bucket_by_adapter``
sub-batches by the (adapters, weights) tuple so each ``set_active_adapters``
call inside a batch window applies to a homogeneous sub-batch.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class LoRAAdapter(BaseModel):
    """A single named LoRA adapter declared on a ``ModelSpec``.

    Args:
        source: Local path or ``hf:org/repo[:weight_file]`` reference.
        weight: Default weight applied when this adapter is selected without
            an explicit ``adapter_weights`` override on the request.  Must be
            non-negative.  Default 1.0.
    """

    source: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0.0)


class LoRAConfig(BaseModel):
    """Adapter registry for a single deployment.

    Args:
        adapters: Mapping of adapter name → :class:`LoRAAdapter`.  Names are
            the user-facing identifiers used in ``request.adapters``.
        default: Optional name from ``adapters`` to apply when the request
            specifies no adapter.  When ``None`` and the request omits
            ``adapters``, no LoRA is applied to that request.

    Raises:
        ValueError: If ``default`` is set but not present in ``adapters``.
    """

    adapters: dict[str, LoRAAdapter] = Field(default_factory=dict)
    default: str | None = None

    @model_validator(mode="after")
    def _validate_default(self) -> LoRAConfig:
        if self.default is not None and self.default not in self.adapters:
            raise ValueError(
                f"LoRAConfig.default={self.default!r} is not in adapters "
                f"(known: {sorted(self.adapters)})"
            )
        return self


def resolve_active_adapters(
    request: Any, lora: LoRAConfig | None
) -> tuple[list[str], list[float]]:
    """Resolve the (names, weights) to activate for *request*.

    Resolution rules:
      - When ``lora`` is ``None``: returns ``([], [])`` — no LoRA applied.
        The caller is responsible for raising if the request itself
        specified adapters but the deployment has none configured.
      - When ``request.adapters`` is non-empty: those names are used.  If
        ``request.adapter_weights`` is set, those weights win; otherwise
        each name's per-adapter ``weight`` from ``lora.adapters`` is used.
      - When ``request.adapters`` is empty and ``lora.default`` is set:
        the default adapter (with its configured weight) is used.
      - Otherwise: ``([], [])``.

    Raises:
        ValueError: If a name in ``request.adapters`` is not registered in
            ``lora.adapters``.
    """
    if lora is None:
        return [], []

    request_adapters = list(getattr(request, "adapters", None) or [])
    if request_adapters:
        unknown = [n for n in request_adapters if n not in lora.adapters]
        if unknown:
            raise ValueError(
                f"Unknown adapter(s) {unknown}; known: {sorted(lora.adapters)}"
            )
        request_weights = getattr(request, "adapter_weights", None)
        if request_weights is None:
            weights = [lora.adapters[n].weight for n in request_adapters]
        else:
            weights = list(request_weights)
        return request_adapters, weights

    if lora.default is not None:
        adapter = lora.adapters[lora.default]
        return [lora.default], [adapter.weight]

    return [], []


def bucket_with_adapter_resolution(
    requests: list[Any], lora: LoRAConfig | None
) -> list[tuple[list[int], list[Any], list[str], list[float]]]:
    """Group *requests* by their resolved (names, weights) adapter selection.

    Returns a list of ``(indices, sub_requests, active_names, active_weights)``
    tuples — one per unique resolved selection, in first-seen order.  Within
    each group, requests are listed in original arrival order.  ``active_names``
    and ``active_weights`` are the resolved adapter set to activate before
    dispatching ``sub_requests`` to the backend.

    Two requests collide on the same key only when they resolve to the
    *same* adapter set in the *same* order with the *same* weights, which
    is the homogeneity guarantee ``set_active_adapters`` requires.

    Raises:
        ValueError: Propagated from :func:`resolve_active_adapters` if any
            request references an unknown adapter.
    """
    buckets: dict[tuple, list[tuple[int, Any]]] = {}
    resolved: dict[tuple, tuple[list[str], list[float]]] = {}
    for i, req in enumerate(requests):
        names, weights = resolve_active_adapters(req, lora)
        key = (tuple(names), tuple(weights))
        buckets.setdefault(key, []).append((i, req))
        resolved.setdefault(key, (names, weights))

    return [
        (
            [i for i, _ in group],
            [r for _, r in group],
            resolved[key][0],
            resolved[key][1],
        )
        for key, group in buckets.items()
    ]


def parse_source(source: str) -> tuple[str, str | None]:
    """Parse a :class:`LoRAAdapter` ``source`` into diffusers load arguments.

    Returns ``(path_or_repo, weight_name)``.  ``weight_name`` is ``None`` for
    local paths and for HF references that don't pin a specific weight file.

    Forms:
        ``"hf:org/repo"``                    → ``("org/repo", None)``
        ``"hf:org/repo:weight.safetensors"`` → ``("org/repo", "weight.safetensors")``
        ``"/abs/path/file.safetensors"``     → ``("/abs/path/file.safetensors", None)``
        ``"./relative/path"``                → ``("./relative/path", None)``
    """
    if not source.startswith("hf:"):
        return source, None
    rest = source[len("hf:") :]
    if ":" in rest:
        repo, weight = rest.split(":", 1)
        return repo, weight
    return rest, None
