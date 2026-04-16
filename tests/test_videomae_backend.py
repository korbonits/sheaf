"""Tests for VideoMAEBackend — mocked, no transformers/GPU required."""

from __future__ import annotations

import base64
import builtins
import struct
import sys
import zlib
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.video import VideoRequest, VideoResponse

# ---------------------------------------------------------------------------
# Minimal PNG frame builder (no PIL needed)
# ---------------------------------------------------------------------------


def _make_frame_b64(width: int = 8, height: int = 8) -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    raw = b"\x00" + b"\xff\xff\xff" * width
    compressed = zlib.compress(raw * height)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _frames(n: int = 16) -> list[str]:
    return [_make_frame_b64() for _ in range(n)]


# ---------------------------------------------------------------------------
# Fake transformers / torch / PIL
# ---------------------------------------------------------------------------


def _make_fake_torch(hidden_dim: int = 768, num_patches: int = 1568) -> MagicMock:
    torch = MagicMock()
    torch.no_grad.return_value.__enter__ = lambda s: None
    torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # Fake hidden state tensor: (1, num_patches+1, hidden_dim)
    hidden = MagicMock()
    hidden.__getitem__ = lambda s, idx: MagicMock(
        norm=lambda **kw: MagicMock(__truediv__=lambda a, b: a),
        cpu=lambda: MagicMock(
            float=lambda: MagicMock(tolist=lambda: [0.0] * hidden_dim)
        ),
        shape=[hidden_dim],
    )
    hidden.mean = lambda dim: hidden
    hidden.__truediv__ = lambda a, b: a

    fake_output = MagicMock()
    fake_output.last_hidden_state = hidden

    # Fake logits: (1, 5) — shape must be a real tuple so shape[-1] returns int
    logits_row = MagicMock()
    fake_output.logits = MagicMock()
    fake_output.logits.shape = (1, 5)
    fake_output.logits.__getitem__ = lambda s, idx: logits_row

    _scores = MagicMock(
        cpu=lambda: MagicMock(
            float=lambda: MagicMock(tolist=lambda: [0.8, 0.1, 0.05, 0.03, 0.02])
        )
    )
    _ids = MagicMock(__iter__=lambda s: iter([0, 1, 2, 3, 4]))
    torch.softmax = MagicMock(return_value=MagicMock())
    torch.topk = MagicMock(return_value=(_scores, _ids))

    return torch, fake_output


def _make_fake_transformers(fake_output: MagicMock) -> MagicMock:
    t = MagicMock()

    model_instance = MagicMock()
    model_instance.return_value = fake_output
    model_instance.eval = MagicMock()
    model_instance.to = MagicMock(return_value=model_instance)
    model_instance.config.id2label = {
        0: "running",
        1: "jumping",
        2: "swimming",
        3: "cycling",
        4: "dancing",
    }

    t.AutoImageProcessor.from_pretrained.return_value = MagicMock()
    t.AutoModel.from_pretrained.return_value = model_instance
    t.AutoModelForVideoClassification.from_pretrained.return_value = model_instance

    return t


def _make_fake_pil() -> MagicMock:
    pil = MagicMock()
    img = MagicMock()
    img.convert.return_value = img
    pil.open.return_value = img
    return pil


def _loaded_backend(
    task: str = "embedding",
    model_name: str = "MCG-NJU/videomae-base",
):
    from sheaf.backends.videomae import VideoMAEBackend

    fake_torch, fake_output = _make_fake_torch()
    fake_transformers = _make_fake_transformers(fake_output)
    fake_pil = _make_fake_pil()

    with patch.dict(
        sys.modules,
        {
            "transformers": fake_transformers,
            "torch": fake_torch,
            "PIL": fake_pil,
            "PIL.Image": fake_pil,
        },
    ):
        backend = VideoMAEBackend(model_name=model_name, task=task, device="cpu")
        backend.load()

    backend._fake_transformers = fake_transformers
    backend._fake_output = fake_output
    backend._fake_torch = fake_torch
    return backend, fake_transformers, fake_torch, fake_output


# ---------------------------------------------------------------------------
# VideoRequest API tests
# ---------------------------------------------------------------------------


class TestVideoRequest:
    def test_defaults(self) -> None:
        req = VideoRequest(model_name="videomae", frames_b64=_frames(16))
        assert req.task == "embedding"
        assert req.pooling == "cls"
        assert req.normalize is True

    def test_classification_task(self) -> None:
        req = VideoRequest(
            model_name="videomae", frames_b64=_frames(8), task="classification"
        )
        assert req.task == "classification"

    def test_rejects_empty_frames(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoRequest(model_name="videomae", frames_b64=[])

    def test_rejects_invalid_base64(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VideoRequest(model_name="videomae", frames_b64=["not-valid-base64!!!"])

    def test_mean_pooling(self) -> None:
        req = VideoRequest(model_name="videomae", frames_b64=_frames(4), pooling="mean")
        assert req.pooling == "mean"


# ---------------------------------------------------------------------------
# VideoMAEBackend unit tests
# ---------------------------------------------------------------------------


class TestVideoMAEBackend:
    def test_load_calls_auto_model(self) -> None:
        from sheaf.backends.videomae import VideoMAEBackend

        fake_torch, fake_output = _make_fake_torch()
        fake_t = _make_fake_transformers(fake_output)
        fake_pil = _make_fake_pil()

        with patch.dict(
            sys.modules,
            {
                "transformers": fake_t,
                "torch": fake_torch,
                "PIL": fake_pil,
                "PIL.Image": fake_pil,
            },
        ):
            backend = VideoMAEBackend(
                model_name="MCG-NJU/videomae-base", task="embedding"
            )
            backend.load()

        fake_t.AutoModel.from_pretrained.assert_called_once_with(
            "MCG-NJU/videomae-base"
        )

    def test_load_uses_classification_model_for_classification_task(self) -> None:
        from sheaf.backends.videomae import VideoMAEBackend

        fake_torch, fake_output = _make_fake_torch()
        fake_t = _make_fake_transformers(fake_output)
        fake_pil = _make_fake_pil()

        with patch.dict(
            sys.modules,
            {
                "transformers": fake_t,
                "torch": fake_torch,
                "PIL": fake_pil,
                "PIL.Image": fake_pil,
            },
        ):
            backend = VideoMAEBackend(
                model_name="MCG-NJU/videomae-base-finetuned-kinetics",
                task="classification",
            )
            backend.load()

        fake_t.AutoModelForVideoClassification.from_pretrained.assert_called_once()
        fake_t.AutoModel.from_pretrained.assert_not_called()

    def test_load_raises_without_transformers(self) -> None:
        _real_import = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return _real_import(name, *args, **kwargs)

        from sheaf.backends.videomae import VideoMAEBackend

        with patch("builtins.__import__", side_effect=_block):
            backend = VideoMAEBackend()
            with pytest.raises(ImportError, match="transformers"):
                backend.load()

    def test_predict_raises_before_load(self) -> None:
        from sheaf.backends.videomae import VideoMAEBackend

        backend = VideoMAEBackend()
        req = VideoRequest(model_name="videomae", frames_b64=_frames())
        with pytest.raises(RuntimeError, match="load()"):
            backend.predict(req)

    def test_predict_returns_video_response(self) -> None:
        backend, *_ = _loaded_backend()
        fake_torch, fake_output = _make_fake_torch()
        req = VideoRequest(model_name="videomae", frames_b64=_frames())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)
        assert isinstance(resp, VideoResponse)
        assert resp.task == "embedding"

    def test_embedding_response_has_dim(self) -> None:
        backend, *_ = _loaded_backend()
        fake_torch, fake_output = _make_fake_torch()
        req = VideoRequest(model_name="videomae", frames_b64=_frames())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)
        assert resp.dim is not None
        assert resp.embedding is not None

    def test_model_type_is_video(self) -> None:
        from sheaf.api.base import ModelType
        from sheaf.backends.videomae import VideoMAEBackend

        assert VideoMAEBackend().model_type == ModelType.VIDEO

    def test_predict_wrong_request_type_raises(self) -> None:
        from sheaf.api.embedding import EmbeddingRequest

        backend, *_ = _loaded_backend()
        req = EmbeddingRequest(model_name="x", texts=["hello"])
        with pytest.raises(TypeError):
            backend.predict(req)

    def test_batch_predict_runs_each(self) -> None:
        backend, *_ = _loaded_backend()
        fake_torch, _ = _make_fake_torch()
        reqs = [
            VideoRequest(model_name="videomae", frames_b64=_frames(), seed=None)
            for _ in range(3)
        ]
        with patch.dict(sys.modules, {"torch": fake_torch}):
            results = backend.batch_predict(reqs)
        assert len(results) == 3
        assert all(isinstance(r, VideoResponse) for r in results)

    def test_processor_called_with_frames(self) -> None:
        backend, fake_t, fake_torch, _ = _loaded_backend()
        req = VideoRequest(model_name="videomae", frames_b64=_frames(4))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            backend.predict(req)
        backend._processor.assert_called_once()
        _, kwargs = backend._processor.call_args
        assert kwargs.get("return_tensors") == "pt"

    def test_classification_response_has_labels(self) -> None:
        backend, *_ = _loaded_backend(task="classification")
        fake_torch, _ = _make_fake_torch()
        req = VideoRequest(
            model_name="videomae-kinetics",
            frames_b64=_frames(),
            task="classification",
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            resp = backend.predict(req)
        assert resp.task == "classification"
        assert resp.labels is not None
        assert resp.scores is not None
        assert len(resp.labels) == len(resp.scores)

    def test_timesformer_model_name(self) -> None:
        backend, fake_t, *_ = _loaded_backend(
            model_name="facebook/timesformer-base-finetuned-k400",
            task="classification",
        )
        fake_t.AutoModelForVideoClassification.from_pretrained.assert_called_once_with(
            "facebook/timesformer-base-finetuned-k400"
        )
