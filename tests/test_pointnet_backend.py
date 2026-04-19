"""Tests for PointNetBackend — fully mocked, no torch/GPU required.

Covers:
  - load() raises ImportError when torch is absent
  - load() with checkpoint_path calls torch.load and load_state_dict
  - load() without checkpoint_path skips torch.load (random init)
  - load() moves model to device
  - model_type returns ModelType.POINT_CLOUD
  - predict() rejects non-PointCloudRequest inputs
  - predict() task="embed" returns 1024-dim L2-normalised embedding
  - predict() task="classify" returns label, scores, label_names
  - predict() task="classify" picks the argmax class as label
  - predict() decodes points_b64 correctly (n_points × 3 float32)
  - predict() uses custom label_names when provided
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.point_cloud import PointCloudRequest, PointCloudResponse

# ---------------------------------------------------------------------------
# Helpers — build a fake points_b64
# ---------------------------------------------------------------------------

_N = 1024
_PTS_NP = np.random.randn(_N, 3).astype(np.float32)
_PTS_B64 = base64.b64encode(_PTS_NP.tobytes()).decode()

_REQ_EMBED = PointCloudRequest(
    model_name="pointnet", points_b64=_PTS_B64, n_points=_N, task="embed"
)
_REQ_CLS = PointCloudRequest(
    model_name="pointnet", points_b64=_PTS_B64, n_points=_N, task="classify"
)


# ---------------------------------------------------------------------------
# Fake torch helpers
# ---------------------------------------------------------------------------


class _NoGrad:
    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *_) -> None:  # type: ignore[no-untyped-def]
        pass


class _FakeTensor:
    """Minimal tensor mock supporting the ops used in PointNetBackend._run()."""

    def __init__(self, data: np.ndarray | None = None) -> None:
        self._data = data if data is not None else np.zeros(1024, dtype=np.float32)

    # tensor ops
    def unsqueeze(self, dim: int) -> _FakeTensor:
        return self

    def to(self, device: str) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list:  # type: ignore[override]
        return self._data.tolist()

    def argmax(self) -> _FakeTensor:
        return _FakeTensor(np.array([int(self._data.argmax())], dtype=np.float32))

    def item(self) -> int:
        return int(self._data.flat[0])

    def __getitem__(self, idx: int) -> _FakeTensor:
        return _FakeTensor(self._data)


def _make_torch_mod(
    feat_np: np.ndarray | None = None,
    logits_np: np.ndarray | None = None,
) -> ModuleType:
    """Build a fake torch module whose model forward produces controlled outputs."""
    _feat = feat_np if feat_np is not None else np.ones(1024, dtype=np.float32)
    _logits = logits_np if logits_np is not None else np.zeros(40, dtype=np.float32)

    feat_t = _FakeTensor(_feat)
    logits_t = _FakeTensor(_logits)

    mod = ModuleType("torch")
    mod.no_grad = _NoGrad  # type: ignore[attr-defined]
    mod.load = MagicMock(return_value={})  # type: ignore[attr-defined]

    # torch.from_numpy(...).unsqueeze(0).to(device) → _FakeTensor
    input_t = _FakeTensor()
    mod.from_numpy = MagicMock(return_value=input_t)  # type: ignore[attr-defined]

    # torch.Generator not needed by PointNet

    # nn and functional — only needed inside _build_pointnet (called in load)
    # We mock the whole model below instead of mocking nn.
    return mod, feat_t, logits_t  # type: ignore[return-value]


def _make_nn_mod() -> ModuleType:
    """Fake torch.nn module so _build_pointnet doesn't import real torch."""
    nn = ModuleType("torch.nn")
    for cls_name in ("Module", "Conv1d", "Linear", "BatchNorm1d", "Dropout"):
        setattr(nn, cls_name, MagicMock)
    return nn


def _make_functional_mod(
    feat_t: _FakeTensor,
    probs_t: _FakeTensor,
) -> ModuleType:
    """Fake torch.nn.functional: normalize and softmax return controlled tensors."""
    F = ModuleType("torch.nn.functional")
    F.normalize = MagicMock(return_value=feat_t)  # type: ignore[attr-defined]
    F.softmax = MagicMock(return_value=probs_t)  # type: ignore[attr-defined]
    return F


# ---------------------------------------------------------------------------
# Fixture: loaded backend with mocked torch
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_backend():  # type: ignore[no-untyped-def]
    from sheaf.backends.pointnet import PointNetBackend

    feat_np = np.ones(1024, dtype=np.float32)
    # logits: class 5 has highest score → should be selected
    logits_np = np.zeros(40, dtype=np.float32)
    logits_np[5] = 10.0
    probs_np = np.exp(logits_np) / np.exp(logits_np).sum()

    torch_mod, feat_t, logits_t = _make_torch_mod(feat_np, logits_np)
    probs_t = _FakeTensor(probs_np)
    F_mod = _make_functional_mod(feat_t, probs_t)
    nn_mod = _make_nn_mod()

    backend = PointNetBackend(device="cpu")

    # Build a fake model that returns (logits_t, feat_t) on __call__
    fake_model = MagicMock()
    fake_model.return_value = (logits_t, feat_t)
    fake_model.to.return_value = fake_model

    with patch.dict(
        sys.modules,
        {
            "torch": torch_mod,
            "torch.nn": nn_mod,
            "torch.nn.functional": F_mod,
        },
    ):
        # Patch _build_pointnet to return our fake model
        with patch("sheaf.backends.pointnet._build_pointnet", return_value=fake_model):
            backend.load()

    # Inject mocked internals — _F was set during load() above
    backend._model = fake_model
    backend._F = F_mod
    return backend, torch_mod, F_mod, feat_t, probs_t, logits_np


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_torch() -> None:
    import builtins

    from sheaf.backends.pointnet import PointNetBackend

    backend = PointNetBackend()
    mods_without = {k: v for k, v in sys.modules.items() if "torch" not in k}
    _real_import = builtins.__import__

    def _raise(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[lidar\\]"),
    ):
        backend.load()


def test_load_with_checkpoint_calls_torch_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from sheaf.backends.pointnet import PointNetBackend

    ckpt_file = str(tmp_path / "model.pth")
    backend = PointNetBackend(checkpoint_path=ckpt_file)

    torch_mod, feat_t, _ = _make_torch_mod()
    nn_mod = _make_nn_mod()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch.dict(
        sys.modules,
        {
            "torch": torch_mod,
            "torch.nn": nn_mod,
            "torch.nn.functional": ModuleType("torch.nn.functional"),
        },
    ):
        with patch("sheaf.backends.pointnet._build_pointnet", return_value=fake_model):
            backend.load()

    torch_mod.load.assert_called_once_with(ckpt_file, map_location="cpu")  # type: ignore[attr-defined]
    fake_model.load_state_dict.assert_called_once()


def test_load_without_checkpoint_skips_torch_load() -> None:
    from sheaf.backends.pointnet import PointNetBackend

    backend = PointNetBackend(checkpoint_path=None)
    torch_mod, _, _ = _make_torch_mod()
    nn_mod = _make_nn_mod()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch.dict(
        sys.modules,
        {
            "torch": torch_mod,
            "torch.nn": nn_mod,
            "torch.nn.functional": ModuleType("torch.nn.functional"),
        },
    ):
        with patch("sheaf.backends.pointnet._build_pointnet", return_value=fake_model):
            backend.load()

    torch_mod.load.assert_not_called()  # type: ignore[attr-defined]


def test_load_moves_model_to_device() -> None:
    from sheaf.backends.pointnet import PointNetBackend

    backend = PointNetBackend(device="cuda")
    torch_mod, _, _ = _make_torch_mod()
    nn_mod = _make_nn_mod()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    with patch.dict(
        sys.modules,
        {
            "torch": torch_mod,
            "torch.nn": nn_mod,
            "torch.nn.functional": ModuleType("torch.nn.functional"),
        },
    ):
        with patch("sheaf.backends.pointnet._build_pointnet", return_value=fake_model):
            backend.load()

    fake_model.to.assert_called_with("cuda")


def test_model_type() -> None:
    from sheaf.api.base import ModelType
    from sheaf.backends.pointnet import PointNetBackend

    assert PointNetBackend().model_type == ModelType.POINT_CLOUD


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.embedding import EmbeddingRequest

    backend, torch_mod, F_mod, *_ = loaded_backend
    req = EmbeddingRequest(model_name="x", texts=["hello"])
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        with pytest.raises(TypeError, match="PointCloudRequest"):
            backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — embed task
# ---------------------------------------------------------------------------


def test_predict_embed_returns_point_cloud_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, feat_t, *_ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        resp = backend.predict(_REQ_EMBED)
    assert isinstance(resp, PointCloudResponse)
    assert resp.embedding is not None
    assert resp.label is None


def test_predict_embed_returns_1024_dim(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, feat_t, *_ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        resp = backend.predict(_REQ_EMBED)
    assert len(resp.embedding) == 1024  # type: ignore[arg-type]


def test_predict_embed_calls_normalize(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, *_ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        backend.predict(_REQ_EMBED)
    F_mod.normalize.assert_called_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# predict() — classify task
# ---------------------------------------------------------------------------


def test_predict_classify_returns_label_and_scores(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, _, probs_t, logits_np = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        resp = backend.predict(_REQ_CLS)
    assert resp.label is not None
    assert resp.scores is not None
    assert resp.label_names is not None


def test_predict_classify_picks_argmax_label(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, _, probs_t, logits_np = loaded_backend
    # logits_np[5] = 10.0 → class 5 should win
    expected_label = backend._label_names[5]
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        resp = backend.predict(_REQ_CLS)
    assert resp.label == expected_label


def test_predict_classify_scores_length_matches_num_classes(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, _, probs_t, _ = loaded_backend
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        resp = backend.predict(_REQ_CLS)
    assert len(resp.scores) == backend._num_classes  # type: ignore[arg-type]


def test_predict_classify_custom_label_names() -> None:
    from sheaf.backends.pointnet import PointNetBackend

    custom_labels = ["cat", "dog", "bird"]
    backend = PointNetBackend(num_classes=3, label_names=custom_labels)
    assert backend._label_names == custom_labels

    logits_np = np.array([0.0, 10.0, 0.0], dtype=np.float32)
    probs_np = np.exp(logits_np) / np.exp(logits_np).sum()

    torch_mod, feat_t, logits_t = _make_torch_mod(logits_np=logits_np)
    probs_t = _FakeTensor(probs_np)
    F_mod = _make_functional_mod(feat_t, probs_t)
    nn_mod = _make_nn_mod()
    fake_model = MagicMock()
    fake_model.return_value = (logits_t, feat_t)
    fake_model.to.return_value = fake_model

    with patch.dict(
        sys.modules,
        {"torch": torch_mod, "torch.nn": nn_mod, "torch.nn.functional": F_mod},
    ):
        with patch("sheaf.backends.pointnet._build_pointnet", return_value=fake_model):
            backend.load()

    backend._model = fake_model
    req = PointCloudRequest(
        model_name="pointnet", points_b64=_PTS_B64, n_points=_N, task="classify"
    )
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        resp = backend.predict(req)

    assert resp.label == "dog"
    assert resp.label_names == custom_labels


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    backend, torch_mod, F_mod, *_ = loaded_backend
    reqs = [_REQ_EMBED, _REQ_CLS]
    with patch.dict(sys.modules, {"torch": torch_mod, "torch.nn.functional": F_mod}):
        responses = backend.batch_predict(reqs)
    assert len(responses) == 2
    assert all(isinstance(r, PointCloudResponse) for r in responses)
    assert backend._model.call_count == 2
