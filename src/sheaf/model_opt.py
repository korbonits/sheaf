"""Inference optimization kits — opt-in per deployment via ``ModelSpec.model_opt``.

Sheaf can engage Anthropic's biomolecular inference optimization kits
(https://github.com/anthropics/uplifting-biomolecular-modeling, Apache-2.0) on
backends that support them.  A kit engages a named **mode**:

- ``off``   — the pinned upstream release exactly as published (the reference
  every other mode is compared against).
- ``exact`` — outputs bitwise identical to ``off``, faster.
- ``fast``  — small, documented numeric differences, faster still.
- ``big``   — lowest peak GPU memory, for inputs the other modes cannot hold.

Each backend declares the modes it supports via
:meth:`~sheaf.backends.base.ModelBackend.supported_opt_modes`.  With
``ModelSpec.model_opt = None`` (the default) nothing of a kit is imported and
the backend runs its pre-existing code path unchanged.

Process isolation
-----------------
A kit patches the upstream package in memory for the *whole interpreter*, and
refuses a second activation under a different mode.  Sheaf therefore requires
one kit mode per process:

- Ray Serve: each replica is its own actor process — safe.
- ``ModalServer``: all specs share one container, so a spec with
  ``model_opt`` must be the only spec in its ``ModalServer``.
- ``BatchRunner``: ``compute="tasks"`` reuses worker processes across specs,
  so ``model_opt`` requires ``compute="actors"``.
- ``SheafWorker``: one spec per process — safe.

:func:`claim_process` is the in-process guard that turns a conflicting second
activation into a clear error instead of a silent mixed-mode process.

Failure contract
----------------
A kit mode either engages or refuses by name (``[<kit>-opt] NOT ACTIVE: …``).
Backends activate the kit explicitly inside ``load()`` (never through the
kit's ``<KIT>_OPT`` environment hook, which ``os._exit(3)``s the process) and
convert a refusal into :class:`ModelOptNotActiveError`, so a Ray Serve
deployment fails at startup with the kit's reason rather than at the first
request.  The kit's ``[<kit>-opt] …`` lines are re-logged through the
``sheaf.model_opt`` logger (see :func:`capture_kit_lines`).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sheaf.backends.base import ModelBackend

_logger = logging.getLogger(__name__)

ModelOptMode = Literal["off", "exact", "fast", "big"]


class ModelOptNotActiveError(RuntimeError):
    """A requested kit mode could not engage (the kit's NOT ACTIVE refusal)."""


class ModelOptConfig(BaseModel):
    """Opt-in inference optimization kit settings for one deployment.

    Attributes:
        mode: Kit mode.  ``off`` runs the upstream release through the kit's
            stock reference path (no kit code imported); ``exact`` / ``fast``
            / ``big`` engage the kit.  Which modes a backend accepts is
            declared by its ``supported_opt_modes()``.
        jit_root: Persistent compile-cache root (the kits'
            ``MODEL_OPT_JIT_ROOT``).  When set, Sheaf exports
            ``TRITON_CACHE_DIR`` and ``TORCH_EXTENSIONS_DIR`` under
            ``<jit_root>/<stack key>/`` before the kit loads, unless they are
            already set.  Must be a directory owned by the serving user and
            not group/other-writable (the kits refuse anything else).
        levers_off: Kit optimizations ("levers") to switch off by name
            (``MODEL_OPT_LEVERS_OFF``).  Only kits that ship ablation
            switches accept it.
    """

    model_config = {"extra": "forbid"}

    mode: ModelOptMode
    jit_root: str | None = None
    levers_off: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Spec-level validation — shared by every serving path
# ---------------------------------------------------------------------------


def apply_model_opt(
    backend: ModelBackend, model_opt: ModelOptConfig | None, deployment: str
) -> None:
    """Hand ``model_opt`` to ``backend`` before ``load()``.

    Raises ``ValueError`` at deploy time when the backend does not support the
    requested mode.  A no-op when ``model_opt`` is ``None``.
    """
    if model_opt is None:
        return
    supported = backend.supported_opt_modes()
    if model_opt.mode not in supported:
        raise ValueError(
            f"ModelSpec '{deployment}' requests model_opt mode "
            f"{model_opt.mode!r} but backend {type(backend).__name__} supports "
            f"{sorted(supported) or 'no inference optimization kit'}."
        )
    backend.configure_model_opt(model_opt)


# ---------------------------------------------------------------------------
# Process-level guard
# ---------------------------------------------------------------------------

# kit name -> (mode, first claimant)
_CLAIMS: dict[str, tuple[str, str]] = {}


def claim_process(kit: str, mode: str, owner: str) -> None:
    """Record that this process runs ``kit`` under ``mode``.

    Kits patch the upstream package process-wide, so a second model of the
    same family in the same process would silently run under the first mode.
    A repeated claim with the same mode is allowed; a different mode raises
    ``ModelOptNotActiveError``.
    """
    prev = _CLAIMS.get(kit)
    if prev is not None and prev[0] != mode:
        raise ModelOptNotActiveError(
            f"{kit} kit already claimed by {prev[1]!r} under mode={prev[0]!r} "
            f"in this process; {owner!r} requests mode={mode!r}. Kits patch "
            "the interpreter process-wide — serve different modes from "
            "separate processes (Ray Serve replicas, separate ModalServer "
            "apps, BatchRunner compute='actors')."
        )
    _CLAIMS.setdefault(kit, (mode, owner))


def _reset_claims() -> None:
    """Test helper: forget all claims in this process."""
    _CLAIMS.clear()


# ---------------------------------------------------------------------------
# Compile cache placement
# ---------------------------------------------------------------------------


def stack_key() -> str:
    """The kits' JIT cache key: ``torch<ver>-cu<cuda>-sm<cc>``.

    Matches ``opt_core.jit_cache`` (e.g. ``torch2.11.0-cu130-sm90``) so a
    cache filled by the kit's own ``run.sh`` is found by Sheaf and vice versa.
    """
    import torch  # ty: ignore[unresolved-import]

    ver = str(torch.__version__).split("+", 1)[0]
    cuda = (torch.version.cuda or "unknown").replace(".", "")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        sm = f"{major}{minor}"
    else:
        sm = "unknown"
    return f"torch{ver}-cu{cuda}-sm{sm}"


def configure_jit_env(model_opt: ModelOptConfig, key: str) -> dict[str, str]:
    """Export the compile-cache variables under ``model_opt.jit_root``.

    Pre-set variables win (operators can override any single directory).
    Returns the variables this call set.  Creates missing directories with
    mode 0700, the private-directory rule the kits enforce.
    """
    if not model_opt.jit_root:
        return {}
    base = os.path.join(model_opt.jit_root, key)
    wanted = {
        "MODEL_OPT_JIT_ROOT": model_opt.jit_root,
        "MODEL_OPT_STACK_KEY": key,
        "TRITON_CACHE_DIR": os.path.join(base, "triton"),
        "TORCH_EXTENSIONS_DIR": os.path.join(base, "torch_extensions"),
    }
    set_now: dict[str, str] = {}
    for name, value in wanted.items():
        if os.environ.get(name):
            continue
        os.environ[name] = value
        set_now[name] = value
    for name in ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        os.makedirs(os.environ[name], mode=0o700, exist_ok=True)
    return set_now


def configure_levers_off(model_opt: ModelOptConfig) -> None:
    """Export ``MODEL_OPT_LEVERS_OFF`` from ``model_opt.levers_off``."""
    if model_opt.levers_off:
        os.environ["MODEL_OPT_LEVERS_OFF"] = ",".join(model_opt.levers_off)


# ---------------------------------------------------------------------------
# Kit line capture
# ---------------------------------------------------------------------------


class _Tee(io.TextIOBase):
    """Writes through to ``inner`` and buffers lines for later inspection.

    Code run inside the capture window (e.g. a logging handler created by a
    first import) can keep a reference to the tee past the block, so after
    :meth:`drain` it is a plain pass-through and stream introspection goes to
    ``inner``.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._buf = ""
        self._capturing = True
        self.lines: list[str] = []

    def write(self, s: str) -> int:
        self._inner.write(s)
        if self._capturing:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self.lines.append(line)
        return len(s)

    def flush(self) -> None:
        self._inner.flush()

    def fileno(self) -> int:
        return self._inner.fileno()

    def isatty(self) -> bool:
        return self._inner.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._inner, "encoding", "utf-8")

    def drain(self) -> None:
        """Stop capturing and flush any partial line into ``lines``."""
        self._capturing = False
        if self._buf:
            self.lines.append(self._buf)
            self._buf = ""


@contextlib.contextmanager
def capture_kit_lines(
    tag: str, deployment: str, sink: list[str] | None = None
) -> Iterator[list[str]]:
    """Capture ``[<tag>] …`` lines a kit prints and re-log them.

    Output still reaches the real stdout/stderr (the kit's own run record is
    kept intact); every line starting with ``[<tag>]`` is additionally logged
    via the ``sheaf.model_opt`` logger with ``kit`` and ``deployment`` extras,
    so the kit's ``ACTIVE mode=…`` line appears in Sheaf's structured logs.

    Yields the list the captured kit lines are appended to (``sink`` if
    given), populated when the block exits.
    """
    lines: list[str] = sink if sink is not None else []
    out, err = _Tee(sys.stdout), _Tee(sys.stderr)
    prefix = f"[{tag}]"
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield lines
    finally:
        for tee in (out, err):
            tee.drain()
            for line in tee.lines:
                if line.startswith(prefix):
                    lines.append(line)
                    _logger.info(line, extra={"kit": tag, "deployment": deployment})
