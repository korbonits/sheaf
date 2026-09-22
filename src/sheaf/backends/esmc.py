"""ESMC backend for protein language modeling via Biohub's esm package.

Requires: pip install "sheaf-serve[protein]"   (Python 3.12+)
Library:  esm (https://github.com/Biohub/esm), released 2026-05-27 under MIT.

Supported models (HuggingFace Hub — weight-downloadable):
  "Biohub/ESMC-6B"  (default) — the only weight-downloadable ESMC variant
                                 as of 2026-05-27.

Forge / Biohub-Platform-only variants (require an API token; not served by
this backend in v0.11 — see docs/adr/0001-esmc-esmfold2-integration.md):
  "esmc-300m-2024-12", "esmc-600m-2024-12"

ESMC follows the standard ``transformers`` masked-LM interface: tokenize a
batch of amino-acid sequences with ``AutoTokenizer``, run a single forward
pass through ``AutoModelForMaskedLM``, and return per-token logits and
optionally per-token hidden-state embeddings. Sequences are length-padded
to the longest in the batch and the attention mask is used to slice
results back to ragged per-sequence lengths before serialisation.

``AutoTokenizer`` and ``AutoModelForMaskedLM`` are stored as instance
attributes at ``load()`` time so the heavy dependency stays lazy and tests
can inject mocks without ``transformers`` or ``torch`` installed.

Inference optimization kit (``ModelSpec.model_opt``)
----------------------------------------------------
With ``model_opt`` set, the backend loads through ``esm``'s own client
instead — ``esm.models.esmc.ESMC.from_pretrained`` (bf16 on GPU, the shipped
varlen flash-attention class) — and runs the padded batched forward
``client.model(input_ids=…, attention_mask=…)`` under bf16 autocast.  That is
the ESM C kit's stock reference ("batched" regime), which the kit's patches
attach to:

- ``mode="off"``   — that stock path, nothing of the kit imported.
- ``mode="exact"`` — ``esmc_opt.enable("exact")`` before the client is built;
  outputs byte-identical to ``off`` for the same batch composition.

The kit requires ``esm`` 3.4.0 at commit ``43ccece2`` with its pinned stack
(torch 2.11.0+cu130, flash-attn 2.7.4.post1, Transformer Engine 2.15.0) — see
``docs/concepts/model_opt.md``.  With ``model_opt=None`` (the default) none of
this runs: the transformers path above is unchanged.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.protein_language import (
    ProteinLanguageRequest,
    ProteinLanguageResponse,
)
from sheaf.backends.base import ModelBackend
from sheaf.model_opt import (
    ModelOptConfig,
    ModelOptNotActiveError,
    capture_kit_lines,
    claim_process,
    configure_jit_env,
    stack_key,
)
from sheaf.registry import register_backend

_DEFAULT_MODEL = "Biohub/ESMC-6B"
_FORGE_MODELS = frozenset({"esmc-300m-2024-12", "esmc-600m-2024-12"})

# model_opt path: accepted model names -> (esm SDK name, ESM C kit variant).
# HF repo IDs are matched case-insensitively (the default is "Biohub/ESMC-6B",
# the SDK's constant is "biohub/ESMC-6B").
_SDK_MODELS = {
    "esmc_300m": ("esmc_300m", "300m"),
    "biohub/esmc-300m": ("esmc_300m", "300m"),
    "esmc_600m": ("esmc_600m", "600m"),
    "biohub/esmc-600m": ("esmc_600m", "600m"),
    "esmc_6b": ("esmc_6b", "6b"),
    "biohub/esmc-6b": ("esmc_6b", "6b"),
}
_KIT = "esmc"
_KIT_TAG = "esmc-opt"
# Human ubiquitin — the kit README's own public warm-up sequence.
_WARMUP_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


@register_backend("esmc")
class ESMCBackend(ModelBackend):
    """ModelBackend for ESMC protein language modeling.

    Args:
        model_name: HuggingFace model ID. Default ``"Biohub/ESMC-6B"`` —
            the only weight-downloadable ESMC variant as of 2026-05-27.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``. The
            6B model effectively requires a GPU with bf16 support; CPU is
            available for testing on smaller dummy weights.
        device_map: When set (e.g. ``"auto"``), passed through to
            ``from_pretrained`` for sharding. Mutually exclusive with
            ``.to(device)`` — when ``device_map`` is set, ``device`` is
            ignored.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        device_map: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._device_map = device_map
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_opt: ModelOptConfig | None = None
        # model_opt path only: the esm SDK client, and the kit's report.
        self._client: Any = None
        self.model_opt_report: dict[str, Any] | None = None
        self.model_opt_lines: list[str] = []

    @property
    def model_type(self) -> str:
        return ModelType.PROTEIN_LANGUAGE

    def supported_opt_modes(self) -> frozenset[str]:
        # The ESM C kit ships off + exact only (no fast / big for this model).
        return frozenset({"off", "exact"})

    def load(self) -> None:
        if self._model_name in _FORGE_MODELS:
            raise NotImplementedError(
                f"ESMC variant {self._model_name!r} is API-only via the "
                "Biohub Platform (Forge) and is not served by this backend "
                "in v0.11. See docs/adr/0001-esmc-esmfold2-integration.md "
                "for the rationale and roadmap."
            )
        if self._model_opt is not None:
            self._load_sdk(self._model_opt)
            return
        try:
            from transformers import (  # ty: ignore[unresolved-import]
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the ESMC backend. "
                "Install it with: pip install 'sheaf-serve[protein]' "
                "(Python 3.12+ required)"
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        kwargs: dict[str, Any] = {}
        if self._device_map is not None:
            kwargs["device_map"] = self._device_map
        self._model = AutoModelForMaskedLM.from_pretrained(self._model_name, **kwargs)
        if self._device_map is None:
            self._model = self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # model_opt path — esm SDK client, optionally under the ESM C kit
    # ------------------------------------------------------------------

    def _load_sdk(self, model_opt: ModelOptConfig) -> None:
        sdk = _SDK_MODELS.get(self._model_name.lower())
        if sdk is None:
            raise ValueError(
                f"model_opt needs one of the ESM C kit's checkpoints; got "
                f"model_name={self._model_name!r}.  Use one of "
                f"{sorted({v[0] for v in _SDK_MODELS.values()})} or the "
                "matching biohub/ESMC-* repo ID."
            )
        sdk_name, variant = sdk
        if self._device_map is not None:
            raise ValueError(
                "model_opt loads through esm's ESMC.from_pretrained onto one "
                "device; device_map is not supported on that path."
            )
        if model_opt.levers_off:
            raise ValueError(
                "The ESM C kit has no switch that drops a lever from a mode "
                "(esmc/CHANGES.md 'Switches'); levers_off must be empty."
            )
        mode = model_opt.mode
        owner = f"esmc:{self._model_name}"
        claim_process(_KIT, mode, owner)

        import torch  # ty: ignore[unresolved-import]

        if mode != "off":
            configure_jit_env(model_opt, stack_key())
            with capture_kit_lines(_KIT_TAG, owner, self.model_opt_lines):
                self.model_opt_report = _enable_kit(mode, variant)

        try:
            from esm.models.esmc import ESMC  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "model_opt on the ESMC backend requires esm 3.4.0 "
                "(git+https://github.com/Biohub/esm.git@"
                "43ccece2ad485f27db46afdb67da2a9601e8f106) with the ESM C "
                "kit's pinned stack.  See docs/concepts/model_opt.md."
            ) from e

        with capture_kit_lines(_KIT_TAG, owner, self.model_opt_lines):
            try:
                client = ESMC.from_pretrained(
                    sdk_name, device=torch.device(self._device)
                )
            except Exception as e:
                # The kit raises its partial-activation refusal here: a lever
                # of the mode could not be applied to the loaded model.
                if _is_kit_refusal(e):
                    raise ModelOptNotActiveError(f"[{_KIT_TAG}] NOT ACTIVE: {e}") from e
                raise
        self._client = client
        self._model = client.model
        self._tokenizer = client.tokenizer
        self._model.eval()

        if mode != "off":
            # One short forward at load: the first request never pays for
            # template recording / Triton warm-up, and a refusal surfaces now.
            with capture_kit_lines(_KIT_TAG, owner, self.model_opt_lines):
                self._run(
                    ProteinLanguageRequest(
                        model_name=self._model_name,
                        sequences=[_WARMUP_SEQUENCE],
                        return_logits=True,
                        return_embeddings=True,
                    )
                )
                self.model_opt_report = _check_kit_status()

    def _autocast(self, torch: Any) -> Any:
        """bf16 autocast on the model_opt path on GPU (the kit's stock call);
        a no-op everywhere else, so the transformers path is unchanged."""
        if self._client is not None and str(self._device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, ProteinLanguageRequest):
            raise TypeError(f"Expected ProteinLanguageRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(  # noqa: C901
        self, request: ProteinLanguageRequest
    ) -> ProteinLanguageResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        inputs = self._tokenizer(
            request.sequences,
            return_tensors="pt",
            padding=True,
        )
        # The model device may differ from self._device when device_map="auto".
        target_device = (
            self._model.device if self._device_map is not None else self._device
        )
        if self._client is not None:
            # model_opt path: exactly the kit's stock batched call — input_ids
            # + attention_mask only, bf16 autocast on GPU.
            inputs = {
                k: inputs[k].to(target_device) for k in ("input_ids", "attention_mask")
            }
        else:
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]  # (N, L)
        seq_lens: list[int] = attention_mask.sum(dim=1).cpu().int().tolist()

        # MaskedLMOutput has .logits + .hidden_states (when requested) but no
        # .last_hidden_state — so requesting embeddings forces the hidden-states
        # flag on the underlying model call.
        need_hidden = request.return_embeddings or request.output_hidden_states
        with torch.inference_mode(), self._autocast(torch):
            output = self._model(
                **inputs,
                output_hidden_states=need_hidden,
            )

        logits_out: list[list[list[float]]] | None = None
        if request.return_logits:
            # logits: (N, L, V) — slice per-sequence by seq_lens[i]
            logits = output.logits.cpu().float()
            logits_out = [
                logits[i, : seq_lens[i], :].tolist() for i in range(len(seq_lens))
            ]

        embeddings_out: list[list[list[float]]] | None = None
        if need_hidden:
            last_hidden = output.hidden_states[-1].cpu().float()
            embeddings_out = [
                last_hidden[i, : seq_lens[i], :].tolist() for i in range(len(seq_lens))
            ]

        hidden_states_out: list[list[list[list[float]]]] | None = None
        if request.output_hidden_states:
            # hidden_states: tuple of (N, L, H), one per layer (including embed)
            hidden_states_out = []
            for layer_hidden in output.hidden_states:
                layer_hidden = layer_hidden.cpu().float()
                hidden_states_out.append(
                    [
                        layer_hidden[i, : seq_lens[i], :].tolist()
                        for i in range(len(seq_lens))
                    ]
                )

        vocab_size = int(output.logits.shape[-1]) if request.return_logits else None
        hidden_dim = None
        if embeddings_out is not None and embeddings_out and embeddings_out[0]:
            hidden_dim = len(embeddings_out[0][0])

        return ProteinLanguageResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            logits=logits_out,
            embeddings=embeddings_out,
            hidden_states=hidden_states_out,
            seq_lens=seq_lens,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
        )


def _enable_kit(mode: str, variant: str) -> dict[str, Any]:
    """``esmc_opt.enable(mode, strict=True)`` → report, or NOT ACTIVE error."""
    try:
        import esmc_opt  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ModelOptNotActiveError(
            f"[{_KIT_TAG}] NOT ACTIVE: the esmc_opt package is not installed "
            f"({e}).  Install the ESM C kit (pip install -e esmc/opt from the "
            "pinned kits tree); see docs/concepts/model_opt.md."
        ) from e
    try:
        # Sheaf always runs the caller's padded batched forward, so the
        # regime is "batched" even for one-sequence requests.
        return esmc_opt.enable(mode, variant=variant, regime="batched", strict=True)
    except esmc_opt.ActivationError as e:
        raise ModelOptNotActiveError(f"[{_KIT_TAG}] NOT ACTIVE: {e}") from e


def _check_kit_status() -> dict[str, Any]:
    """After the warm-up forward: the kit must report every lever applied."""
    import esmc_opt  # ty: ignore[unresolved-import]

    report = esmc_opt.status()
    fallback = report.get("levers_fallback") or []
    if not report.get("active") or fallback:
        raise ModelOptNotActiveError(
            f"[{_KIT_TAG}] NOT ACTIVE: mode={report.get('mode')} "
            f"levers_fallback={fallback} reason={report.get('reason')}"
        )
    return report


def _is_kit_refusal(exc: BaseException) -> bool:
    """True for the kit's own ActivationError family (esmc_opt or its stack)."""
    for mod_name in ("esmc_opt", "esmc_opt.stack"):
        mod = sys.modules.get(mod_name)
        err = getattr(mod, "ActivationError", None) if mod is not None else None
        if isinstance(err, type) and isinstance(exc, err):
            return True
    return False
