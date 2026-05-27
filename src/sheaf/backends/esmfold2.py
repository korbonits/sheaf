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
"""

from __future__ import annotations

from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.structure import StructureRequest, StructureResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_DEFAULT_MODEL = "biohub/ESMFold2"
_FORGE_MODELS = frozenset({"esmfold2-fast-2026-05"})


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

    @property
    def model_type(self) -> str:
        return ModelType.STRUCTURE

    def load(self) -> None:
        if self._model_name in _FORGE_MODELS:
            raise NotImplementedError(
                f"ESMFold2 variant {self._model_name!r} is API-only via "
                "the Biohub Platform (Forge) and is not served by this "
                "backend in v0.11. See "
                "docs/adr/0001-esmc-esmfold2-integration.md for the "
                "rationale and roadmap."
            )
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

        result = self._InputBuilder().fold(
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
