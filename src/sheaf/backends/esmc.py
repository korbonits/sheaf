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
"""

from __future__ import annotations

from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.protein_language import (
    ProteinLanguageRequest,
    ProteinLanguageResponse,
)
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

_DEFAULT_MODEL = "Biohub/ESMC-6B"
_FORGE_MODELS = frozenset({"esmc-300m-2024-12", "esmc-600m-2024-12"})


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

    @property
    def model_type(self) -> str:
        return ModelType.PROTEIN_LANGUAGE

    def load(self) -> None:
        if self._model_name in _FORGE_MODELS:
            raise NotImplementedError(
                f"ESMC variant {self._model_name!r} is API-only via the "
                "Biohub Platform (Forge) and is not served by this backend "
                "in v0.11. See docs/adr/0001-esmc-esmfold2-integration.md "
                "for the rationale and roadmap."
            )
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
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]  # (N, L)
        seq_lens: list[int] = attention_mask.sum(dim=1).cpu().int().tolist()

        # MaskedLMOutput has .logits + .hidden_states (when requested) but no
        # .last_hidden_state — so requesting embeddings forces the hidden-states
        # flag on the underlying model call.
        need_hidden = request.return_embeddings or request.output_hidden_states
        with torch.inference_mode():
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
