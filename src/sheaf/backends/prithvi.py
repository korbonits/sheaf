"""Prithvi Earth Observation backend via HuggingFace transformers.

Requires: pip install "sheaf-serve[earth-observation]"
Library:  transformers (trust_remote_code=True for Prithvi custom classes)

Supported models (HuggingFace Hub):
  "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"    — 100M params, 6 HLS bands, T=3
  "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"    — 300M params, flexible bands/time
  "ibm-nasa-geospatial/Prithvi-EO-2.0-600M"    — 600M params

Prithvi is a masked-autoencoder (MAE) ViT trained on Harmonized Landsat
Sentinel-2 (HLS) data at 30 m resolution.  It is an image-only backbone;
there is no text encoder.

Input convention
----------------
``SatelliteRequest.pixels_b64`` must encode a float32 array of shape
(n_time, n_bands, height, width).  For Prithvi EO 1.0 the expected shape
is (3, 6, 224, 224); for EO 2.0 the temporal and channel dimensions are
flexible.

Standard HLS band order (must match the model's training bands):
  ["blue", "green", "red", "nir08", "swir16", "swir22"]

Normalization
-------------
When ``SatelliteRequest.normalize=True`` (default), per-band z-score
normalization is applied using the ``image_mean`` and ``image_std``
statistics stored in the model's ``AutoImageProcessor`` config.  These
stats match the HLS training distribution.  If your data is already
normalized, set ``normalize=False``.

Pooling
-------
Prithvi's MAE does not use a CLS token, so ``pooling="mean"`` (default)
is recommended: all output tokens are averaged into a single vector.
``pooling="cls"`` takes the first token (index 0); useful if the model
version you are using prepends register / CLS tokens.

Output
------
Returns a ``SatelliteResponse`` with a scene-level embedding of shape
``(dim,)`` where ``dim`` is the model's hidden size (e.g. 768 for 300M).
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.satellite import SatelliteRequest, SatelliteResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("prithvi")
class PrithviBackend(ModelBackend):
    """ModelBackend for Prithvi Earth Observation embeddings.

    Loads a Prithvi model checkpoint from HuggingFace Hub, applies
    per-band normalization (optional), runs a forward pass, and returns
    a mean-pooled or first-token scene embedding.

    Args:
        model_name: HuggingFace model ID.
            "ibm-nasa-geospatial/Prithvi-EO-2.0-300M" (default)
        device: "cpu", "cuda", or "mps"
    """

    def __init__(
        self,
        model_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None  # AutoImageProcessor, stored for testability

    @property
    def model_type(self) -> str:
        return ModelType.GEOSPATIAL

    def load(self) -> None:
        try:
            from transformers import (  # ty: ignore[unresolved-import]
                AutoImageProcessor,
                AutoModel,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is required for the Prithvi backend. "
                "Install it with: pip install 'sheaf-serve[earth-observation]'"
            ) from e

        self._processor = AutoImageProcessor.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, SatelliteRequest):
            raise TypeError(f"Expected SatelliteRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _run(self, request: SatelliteRequest) -> SatelliteResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        # Decode: (n_time, n_bands, H, W)
        pixels = np.frombuffer(
            base64.b64decode(request.pixels_b64), dtype=np.float32
        ).reshape(request.n_time, request.n_bands, request.height, request.width)

        if request.normalize:
            pixels = self._normalize(pixels)

        # Add batch dim → (1, n_time, n_bands, H, W)
        pixel_tensor = torch.from_numpy(pixels).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_tensor)

        # last_hidden_state: (1, seq_len, hidden_size)
        hidden = outputs.last_hidden_state[0]  # (seq_len, hidden_size)

        if request.pooling == "cls":
            emb = hidden[0]  # first token
        else:
            emb = hidden.mean(dim=0)  # mean over all tokens

        return SatelliteResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embedding=emb.cpu().float().tolist(),
            dim=int(emb.shape[0]),
            n_time=request.n_time,
        )

    def _normalize(self, pixels: np.ndarray) -> np.ndarray:
        """Per-band z-score using image_mean / image_std from processor config.

        Falls back to identity if the processor doesn't expose those stats.
        pixels shape: (n_time, n_bands, H, W)
        """
        try:
            means = np.array(self._processor.image_mean, dtype=np.float32)
            stds = np.array(self._processor.image_std, dtype=np.float32)
        except (AttributeError, TypeError):
            return pixels  # no stats available — pass through

        if means.shape[0] != pixels.shape[1]:
            # Mismatch between processor bands and request bands — skip.
            return pixels

        # Broadcast over (n_time, n_bands, H, W)
        means = means[np.newaxis, :, np.newaxis, np.newaxis]
        stds = stds[np.newaxis, :, np.newaxis, np.newaxis]
        stds = np.where(stds > 0, stds, 1.0)  # avoid division by zero
        return (pixels - means) / stds
