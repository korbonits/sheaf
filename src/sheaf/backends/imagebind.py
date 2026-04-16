"""ImageBind backend for cross-modal embeddings (text, vision, audio, depth, thermal).

Requires: pip install "sheaf-serve[multimodal]"
Model: imagebind_huge (1.2B params, 1024-dim shared embedding space)

ImageBind maps six modalities into a single embedding space, enabling
cross-modal retrieval (e.g. find images matching an audio clip).  This
backend supports five modalities: text, vision, audio, depth, and thermal.

Modality notes:
  text    — plain strings, no temp files needed.
  vision  — base64-encoded image files (JPEG/PNG/WebP).
  audio   — base64-encoded audio files (WAV/MP3); 2-second clips at 16kHz.
  depth   — base64-encoded depth images (single-channel or 3-channel).
  thermal — base64-encoded thermal images (grayscale or colour).

Image/audio inputs are written to named temp files because the ImageBind
data loaders operate on file paths, not in-memory buffers.  Temp files are
cleaned up in a finally block after inference.

self._ModalityType is stored at load() time so unit tests can inject a mock
object without the imagebind package installed in the test environment.
"""

from __future__ import annotations

import base64
import os
import tempfile
from typing import Any

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.multimodal_embedding import (
    MODALITY_AUDIO,
    MODALITY_DEPTH,
    MODALITY_TEXT,
    MODALITY_THERMAL,
    MODALITY_VISION,
    MultimodalEmbeddingRequest,
    MultimodalEmbeddingResponse,
)
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend

# Suffix map: modality → temp-file extension used when writing bytes to disk.
_SUFFIX: dict[str, str] = {
    MODALITY_VISION: ".jpg",
    MODALITY_AUDIO: ".wav",
    MODALITY_DEPTH: ".png",
    MODALITY_THERMAL: ".jpg",
}


def _write_temp_files(items_b64: list[str], suffix: str) -> list[str]:
    """Decode base64 items and write each to a named temp file.

    Returns a list of file paths.  The caller is responsible for deleting
    them (use in a try/finally block).
    """
    paths: list[str] = []
    for b64 in items_b64:
        data = base64.b64decode(b64)
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception:
            os.close(fd)
            raise
        paths.append(path)
    return paths


def _remove_files(paths: list[str]) -> None:
    """Best-effort cleanup of temp files."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


@register_backend("imagebind")
class ImageBindBackend(ModelBackend):
    """ModelBackend for ImageBind cross-modal embeddings.

    Accepts one modality per request (text, vision, audio, depth, or thermal)
    and returns 1024-dim vectors in ImageBind's shared embedding space.
    All embeddings are L2-normalized by default — cosine similarity equals
    dot product.

    Args:
        pretrained: If True (default), download ImageBind weights from the
            EvolutionaryScale / Meta checkpoint.  Set to False for fast tests
            that inject a mock model.
        device: "cpu", "cuda", or "mps".
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        self._pretrained = pretrained
        self._device = device
        self._model: Any = None
        self._data: Any = None  # imagebind.data module, stored for testability
        self._ModalityType: Any = None  # imagebind ModalityType, stored for testability

    @property
    def model_type(self) -> str:
        return ModelType.MULTIMODAL_EMBEDDING

    def load(self) -> None:
        try:
            import imagebind.data as _data  # ty: ignore[unresolved-import]
            from imagebind.models import (  # ty: ignore[unresolved-import]
                imagebind_model,
            )
            from imagebind.models.imagebind_model import (  # ty: ignore[unresolved-import]
                ModalityType,
            )
        except ImportError as e:
            raise ImportError(
                "imagebind is required for the ImageBind backend. "
                "Install it from the official Meta repo: "
                "pip install git+https://github.com/facebookresearch/ImageBind.git "
                "then: pip install 'sheaf-serve[multimodal]'"
            ) from e

        self._data = _data
        self._ModalityType = ModalityType
        self._model = imagebind_model.imagebind_huge(pretrained=self._pretrained)
        self._model.eval()
        self._model = self._model.to(self._device)

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, MultimodalEmbeddingRequest):
            raise TypeError(f"Expected MultimodalEmbeddingRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: MultimodalEmbeddingRequest) -> MultimodalEmbeddingResponse:
        import torch  # ty: ignore[unresolved-import]

        if self._model is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        modality = request.modality
        temp_paths: list[str] = []

        try:
            inputs = self._build_inputs(request, modality, temp_paths)

            with torch.no_grad():
                embeddings_dict = self._model(inputs)

            # Retrieve the tensor for the active modality key.
            modality_key = self._modality_key(modality)
            embs = embeddings_dict[modality_key]  # (N, 1024)

            if request.normalize:
                embs = embs / embs.norm(dim=-1, keepdim=True)

            embs_list: list[list[float]] = embs.cpu().float().tolist()

        finally:
            _remove_files(temp_paths)

        return MultimodalEmbeddingResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            embeddings=embs_list,
            dim=embs.shape[-1],
            modality=modality,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _modality_key(self, modality: str) -> Any:
        """Return the ModalityType enum value for the given modality string."""
        mt = self._ModalityType
        return {
            MODALITY_TEXT: mt.TEXT,
            MODALITY_VISION: mt.VISION,
            MODALITY_AUDIO: mt.AUDIO,
            MODALITY_DEPTH: mt.DEPTH,
            MODALITY_THERMAL: mt.THERMAL,
        }[modality]

    def _build_inputs(
        self,
        request: MultimodalEmbeddingRequest,
        modality: str,
        temp_paths: list[str],
    ) -> dict[Any, Any]:
        """Build the inputs dict expected by imagebind's forward pass."""
        mt = self._ModalityType
        d = self._data

        if modality == MODALITY_TEXT:
            assert request.texts is not None
            return {mt.TEXT: d.load_and_transform_text(request.texts, self._device)}

        if modality == MODALITY_VISION:
            assert request.images_b64 is not None
            paths = _write_temp_files(request.images_b64, _SUFFIX[modality])
            temp_paths.extend(paths)
            return {mt.VISION: d.load_and_transform_vision_data(paths, self._device)}

        if modality == MODALITY_AUDIO:
            assert request.audios_b64 is not None
            paths = _write_temp_files(request.audios_b64, _SUFFIX[modality])
            temp_paths.extend(paths)
            return {mt.AUDIO: d.load_and_transform_audio_data(paths, self._device)}

        if modality == MODALITY_DEPTH:
            assert request.depth_images_b64 is not None
            paths = _write_temp_files(request.depth_images_b64, _SUFFIX[modality])
            temp_paths.extend(paths)
            return {mt.DEPTH: d.load_and_transform_depth_data(paths, self._device)}

        if modality == MODALITY_THERMAL:
            assert request.thermal_images_b64 is not None
            paths = _write_temp_files(request.thermal_images_b64, _SUFFIX[modality])
            temp_paths.extend(paths)
            return {mt.THERMAL: d.load_and_transform_thermal_data(paths, self._device)}

        raise ValueError(f"Unsupported modality: {modality}")
