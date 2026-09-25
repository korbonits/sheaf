"""ESMFold2 backend for protein structure prediction via Biohub's esm package.

Requires: pip install "sheaf-serve[protein]"   (Python 3.12+)
Library:  esm (https://github.com/Biohub/esm), released 2026-05-27 under MIT.

Supported models (HuggingFace Hub — weight-downloadable):
  "biohub/ESMFold2"  (default) — note lowercase ``biohub/``, mirroring the
                                  upstream README verbatim.

Forge / Biohub-Platform-only variants (require an API token; not served by
this backend in v0.11 — see docs/adr/0001-esmc-esmfold2-integration.md):
  "esmfold2-fast-2026-05"

ESMFold2 wraps a 6B-parameter ESMC language model with a diffusion-based
structure-prediction head. The headline capability is *inference-time
scaling*: the looped-transformer recurrence (``num_loops``), the diffusion
sampler (``num_sampling_steps``), and the number of independent samples
(``num_diffusion_samples``) trade compute for accuracy at predict time.
We surface all three plus the random ``seed`` as first-class fields on
:class:`~sheaf.api.structure.StructureRequest`.

Structure prediction is **single-sample-per-call** at the upstream API
level; ``batch_predict`` runs requests sequentially. Per-request compute
varies hugely with sequence length × num_loops × num_samples; do not
expect uniform latency.

Inference optimization kit (``ModelSpec.model_opt``)
----------------------------------------------------
With ``model_opt`` set, the backend runs the ESMFold2 kit's pinned stack
(esm 3.3.0 @ ``26b0bc2b``, from the kit image): ``off`` is the library's
fused backend with chunking off (``set_kernel_backend("fused")`` +
``set_chunk_size(None)``), the configuration ``exact`` reproduces bit for bit
under the kit's deterministic recipe; ``exact`` and ``fast`` call
``esmfold2_opt.enable(mode, strict=True)`` before the model is built, apply the
kit to it at load (``stack.apply_to``) and refuse at startup when any lever of
the mode is left unapplied.  ``biohub/ESMFold2`` runs as the kit's
``full_nomsa`` variant and ``biohub/ESMFold2-Fast`` as ``fast``; one mode and
one variant per process.  See ``docs/concepts/model_opt.md``.  With
``model_opt=None`` (the default) none of this runs.
"""

from __future__ import annotations

from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.structure import ChainInput, StructureRequest, StructureResponse
from sheaf.backends.base import ModelBackend
from sheaf.model_opt import (
    ModelOptConfig,
    ModelOptNotActiveError,
    capture_kit_lines,
    claim_process,
    configure_jit_env,
    configure_levers_off,
    stack_key,
)
from sheaf.registry import register_backend

_DEFAULT_MODEL = "biohub/ESMFold2"
_FORGE_MODELS = frozenset({"esmfold2-fast-2026-05"})

# model_opt path: accepted HF repo IDs (case-insensitive) -> ESMFold2 kit variant.
_KIT_VARIANTS = {
    "biohub/esmfold2": "full_nomsa",
    "biohub/esmfold2-fast": "fast",
}
_KIT = "esmfold2"
_KIT_TAG = "esmfold2-opt"
# Human ubiquitin (P0CG48) — a small public warm-up input.
_WARMUP_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


@register_backend("esmfold2")
class ESMFold2Backend(ModelBackend):
    """ModelBackend for ESMFold2 protein structure prediction.

    Args:
        model_name: HuggingFace model ID. Default ``"biohub/ESMFold2"``.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:N"``. The 6B-backed
            structure head effectively requires a GPU with bf16 support
            for practical inference latencies.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        # Stored at load() for test injectability — same pattern as
        # ESM3Backend._ESMProtein, OpenCLIPBackend._Image, etc.
        self._ProteinInput: Any = None
        self._StructurePredictionInput: Any = None
        self._InputBuilder: Any = None
        # model_opt path only: one builder for the process, because the kit's
        # feature-cache lever lives on the builder it was applied with.
        self._builder: Any = None
        self._model_opt: ModelOptConfig | None = None
        self.model_opt_report: dict[str, Any] | None = None
        self.model_opt_lines: list[str] = []

    @property
    def model_type(self) -> str:
        return ModelType.STRUCTURE

    def supported_opt_modes(self) -> frozenset[str]:
        # "big" (low-memory, large inputs) is not wired up yet.
        return frozenset({"off", "exact", "fast"})

    def load(self) -> None:
        if self._model_name in _FORGE_MODELS:
            raise NotImplementedError(
                f"ESMFold2 variant {self._model_name!r} is API-only via "
                "the Biohub Platform (Forge) and is not served by this "
                "backend in v0.11. See "
                "docs/adr/0001-esmc-esmfold2-integration.md for the "
                "rationale and roadmap."
            )
        if self._model_opt is not None:
            self._load_opt(self._model_opt)
            return
        try:
            from esm.models.esmfold2 import (  # ty: ignore[unresolved-import]
                ESMFold2InputBuilder,
                ProteinInput,
                StructurePredictionInput,
            )
            from transformers.models.esmfold2.modeling_esmfold2 import (  # ty: ignore[unresolved-import]
                ESMFold2Model,
            )
        except ImportError as e:
            raise ImportError(
                "esm and transformers are required for the ESMFold2 backend. "
                "Install them with: pip install 'sheaf-serve[protein]' "
                "(Python 3.12+ required)"
            ) from e

        self._model = ESMFold2Model.from_pretrained(self._model_name)
        # ``.cuda()`` and ``.eval()`` mirror the upstream README example.
        # We call ``.to(device)`` instead so CPU testing works on a stub.
        self._model = self._model.to(self._device)
        self._model.eval()
        self._ProteinInput = ProteinInput
        self._StructurePredictionInput = StructurePredictionInput
        self._InputBuilder = ESMFold2InputBuilder

    # ------------------------------------------------------------------
    # model_opt path — the ESMFold2 kit's pinned stack
    # ------------------------------------------------------------------

    def _load_opt(self, model_opt: ModelOptConfig) -> None:
        variant = _KIT_VARIANTS.get(self._model_name.lower())
        if variant is None:
            raise ValueError(
                f"model_opt needs one of the ESMFold2 kit's checkpoints; got "
                f"model_name={self._model_name!r}.  Use biohub/ESMFold2 "
                "(kit variant full_nomsa) or biohub/ESMFold2-Fast (fast)."
            )
        mode = model_opt.mode
        if mode != "off" and not str(self._device).startswith("cuda"):
            raise ValueError(
                f"model_opt mode {mode!r} needs a CUDA device (the ESMFold2 "
                f"kit's levers are GPU kernels); got device={self._device!r}."
            )
        owner = f"esmfold2:{self._model_name}"
        claim_process(_KIT, f"{mode}/{variant}", owner)

        try:
            from esm.models.esmfold2 import (  # ty: ignore[unresolved-import]
                ESMFold2InputBuilder,
                ProteinInput,
                StructurePredictionInput,
            )
            from transformers.models.esmfold2.modeling_esmfold2 import (  # ty: ignore[unresolved-import]
                ESMFold2Model,
            )
        except ImportError as e:
            raise ImportError(
                "model_opt on the ESMFold2 backend requires esm 3.3.0 "
                "(git+https://github.com/Biohub/esm.git@"
                "26b0bc2b) with the ESMFold2 kit's pinned stack.  See "
                "docs/concepts/model_opt.md."
            ) from e

        if mode != "off":
            configure_jit_env(model_opt, stack_key())
            configure_levers_off(model_opt)
            with capture_kit_lines(_KIT_TAG, owner, self.model_opt_lines):
                self.model_opt_report = _enable_kit(mode, variant)

        model = ESMFold2Model.from_pretrained(self._model_name)
        model = model.to(self._device)
        model.eval()
        builder = ESMFold2InputBuilder()
        if mode == "off":
            # The kit's reference for exact (stock_fold.BACKENDS["fused"]).
            model.set_kernel_backend("fused")
            model.set_chunk_size(None)
        else:
            with capture_kit_lines(_KIT_TAG, owner, self.model_opt_lines):
                _apply_kit(model, builder)

        self._model = model
        self._builder = builder
        self._ProteinInput = ProteinInput
        self._StructurePredictionInput = StructurePredictionInput
        self._InputBuilder = ESMFold2InputBuilder

        if mode != "off":
            # One fold at load: CUDA-graph capture and Triton compiles happen
            # here, not on the first request, and a runtime refusal surfaces now.
            with capture_kit_lines(_KIT_TAG, owner, self.model_opt_lines):
                self._run(
                    StructureRequest(
                        model_name=self._model_name,
                        chains=[ChainInput(chain_id="A", sequence=_WARMUP_SEQUENCE)],
                    )
                )
                self.model_opt_report = _check_kit_status()

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, StructureRequest):
            raise TypeError(f"Expected StructureRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        # Structure prediction is single-sample-per-call upstream; sequential.
        return [self.predict(r) for r in requests]

    def _run(self, request: StructureRequest) -> StructureResponse:
        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        protein_inputs = [
            self._ProteinInput(id=c.chain_id, sequence=c.sequence)
            for c in request.chains
        ]
        spi = self._StructurePredictionInput(sequences=protein_inputs)

        builder = self._builder if self._builder is not None else self._InputBuilder()
        result = builder.fold(
            self._model,
            spi,
            num_loops=request.num_loops,
            num_sampling_steps=request.num_sampling_steps,
            num_diffusion_samples=request.num_samples,
            seed=request.seed,
        )

        if request.output_format == "pdb":
            structure_str = result.complex.to_pdb()
        else:
            structure_str = result.complex.to_mmcif()

        # pLDDT — coerce to a flat list[float]. Upstream returns a tensor;
        # we go via .cpu().float().tolist() if it's a tensor, or trust the
        # value as-is if it's already a list (test stubs).
        plddt = _to_float_list(result.plddt)

        ptm = _maybe_float(getattr(result, "ptm", None))
        iptm = _maybe_float(getattr(result, "iptm", None))
        pae = _maybe_2d_list(getattr(result, "pae", None))
        sample_scores = (
            _to_float_list(getattr(result, "sample_scores", None))
            if request.num_samples > 1
            and getattr(result, "sample_scores", None) is not None
            else None
        )

        return StructureResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            structure=structure_str,
            structure_format=request.output_format,
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            pae=pae,
            sample_scores=sample_scores,
        )


def _enable_kit(mode: str, variant: str) -> dict[str, Any]:
    """``esmfold2_opt.enable(mode, variant, strict=True)`` → report, or NOT ACTIVE."""
    try:
        import esmfold2_opt  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ModelOptNotActiveError(
            f"[{_KIT_TAG}] NOT ACTIVE: the esmfold2_opt package is not "
            f"installed ({e}).  Use the ESMFold2 kit image; see "
            "docs/concepts/model_opt.md."
        ) from e
    try:
        return esmfold2_opt.enable(mode, variant=variant, strict=True)
    except esmfold2_opt.ActivationError as e:
        raise ModelOptNotActiveError(f"[{_KIT_TAG}] NOT ACTIVE: {e}") from e


def _apply_kit(model: Any, builder: Any) -> None:
    """Install the active mode's levers on ``model`` (``stack.apply_to``)."""
    import esmfold2_opt  # ty: ignore[unresolved-import]

    try:
        esmfold2_opt.stack.apply_to(model, builder)
    except esmfold2_opt.ActivationError as e:
        raise ModelOptNotActiveError(f"[{_KIT_TAG}] NOT ACTIVE: {e}") from e
    except SystemExit as e:
        # apply_to exits 3 when the flash-attention / Transformer Engine paths
        # are not live under ESMFOLD2_OPT_REQUIRE_FAST_ENV=1; a Ray replica
        # must fail its startup instead of the process exiting.
        raise ModelOptNotActiveError(
            f"[{_KIT_TAG}] NOT ACTIVE: the kit refused the loaded model "
            f"(exit {e.code}); its NOT ACTIVE line above gives the reason."
        ) from None


def _check_kit_status() -> dict[str, Any]:
    """After the warm-up fold: every lever of the mode must be applied."""
    import esmfold2_opt  # ty: ignore[unresolved-import]

    report = esmfold2_opt.status()
    partial = report.get("partial") or []
    if not report.get("active") or partial:
        reasons = report.get("fallback_reasons") or {}
        raise ModelOptNotActiveError(
            f"[{_KIT_TAG}] NOT ACTIVE: mode={report.get('mode')} "
            f"variant={report.get('variant')} partial={partial} "
            f"reasons={reasons} reason={report.get('reason')}"
        )
    return report


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu().float().tolist()
    if isinstance(value, list):
        return [float(x) for x in value]
    return [float(value)]


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _maybe_2d_list(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu().float().tolist()
    return [[float(x) for x in row] for row in value]
