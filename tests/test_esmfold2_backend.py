"""Tests for ESMFold2Backend — fully mocked, no esm or transformers required.

Covers:
  - load() raises ImportError when esm/transformers absent
  - load() rejects Forge-only model IDs with NotImplementedError
  - load() passes model_name through and moves model to device
  - load() stores ProteinInput, StructurePredictionInput, InputBuilder for injection
  - predict() rejects non-StructureRequest inputs
  - predict() builds a ProteinInput per chain and passes them into spi.sequences
  - predict() threads num_loops / num_sampling_steps / num_samples / seed
    through to ESMFold2InputBuilder().fold()
  - predict() returns StructureResponse with structure string in the requested format
  - predict() pdb vs mmcif output format selects the correct serialiser
  - predict() copies plddt, ptm, iptm and pae onto the response
  - predict() returns sample_scores only when num_samples > 1
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sheaf.api.structure import ChainInput, StructureRequest, StructureResponse

# ---------------------------------------------------------------------------
# Fake esm.models.esmfold2 module
# ---------------------------------------------------------------------------


class _FakeProteinInput:
    def __init__(self, id: str, sequence: str) -> None:
        self.id = id
        self.sequence = sequence


class _FakeStructurePredictionInput:
    def __init__(self, sequences: list[_FakeProteinInput]) -> None:
        self.sequences = sequences


class _FakeComplex:
    def __init__(self, mmcif: str = "CIF-BODY", pdb: str = "PDB-BODY") -> None:
        self._mmcif = mmcif
        self._pdb = pdb

    def to_mmcif(self) -> str:
        return self._mmcif

    def to_pdb(self) -> str:
        return self._pdb


class _FakeFloat:
    """Minimal stand-in for torch scalars with .item()."""

    def __init__(self, value: float) -> None:
        self._v = value

    def item(self) -> float:
        return self._v


class _FakeListTensor:
    """Mimics a torch tensor that supports .cpu().float().tolist()."""

    def __init__(self, data: list) -> None:
        self._data = data

    def cpu(self) -> _FakeListTensor:
        return self

    def float(self) -> _FakeListTensor:
        return self

    def tolist(self) -> list:
        return self._data


def _make_result(
    plddt: list[float] | None = None,
    ptm: float | None = 0.91,
    iptm: float | None = 0.42,
    pae: list[list[float]] | None = None,
    sample_scores: list[float] | None = None,
    mmcif: str = "CIF-BODY",
    pdb: str = "PDB-BODY",
) -> Any:
    if plddt is None:
        plddt = [85.0, 90.0, 75.0]
    res = MagicMock()
    res.plddt = _FakeListTensor(plddt)
    res.ptm = _FakeFloat(ptm) if ptm is not None else None
    res.iptm = _FakeFloat(iptm) if iptm is not None else None
    res.pae = _FakeListTensor(pae) if pae is not None else None
    res.sample_scores = (
        _FakeListTensor(sample_scores) if sample_scores is not None else None
    )
    res.complex = _FakeComplex(mmcif=mmcif, pdb=pdb)
    return res


class _FakeInputBuilder:
    """Captures kwargs from .fold() so tests can assert on them."""

    last_call_kwargs: dict[str, Any] | None = None
    result: Any = None

    def fold(self, model: Any, spi: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        _FakeInputBuilder.last_call_kwargs = kwargs
        return _FakeInputBuilder.result


def _make_esm_mod() -> ModuleType:
    esm_mod = ModuleType("esm")
    models_mod = ModuleType("esm.models")
    esmfold2_mod = ModuleType("esm.models.esmfold2")

    esmfold2_mod.ESMFold2InputBuilder = _FakeInputBuilder  # type: ignore[attr-defined]
    esmfold2_mod.ProteinInput = _FakeProteinInput  # type: ignore[attr-defined]
    esmfold2_mod.StructurePredictionInput = _FakeStructurePredictionInput  # type: ignore[attr-defined]

    esm_mod.models = models_mod  # type: ignore[attr-defined]
    models_mod.esmfold2 = esmfold2_mod  # type: ignore[attr-defined]

    return esm_mod


def _esm_sys_modules(esm_mod: ModuleType) -> dict[str, ModuleType]:
    return {
        "esm": esm_mod,
        "esm.models": esm_mod.models,  # type: ignore[attr-defined]
        "esm.models.esmfold2": esm_mod.models.esmfold2,  # type: ignore[attr-defined]
    }


# ---------------------------------------------------------------------------
# Fake transformers module
# ---------------------------------------------------------------------------


def _make_transformers_mod() -> tuple[ModuleType, MagicMock]:
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = None

    transformers_mod = ModuleType("transformers")
    models_mod = ModuleType("transformers.models")
    esmfold2_mod = ModuleType("transformers.models.esmfold2")
    modeling_mod = ModuleType("transformers.models.esmfold2.modeling_esmfold2")

    esmfold2_cls = MagicMock()
    esmfold2_cls.from_pretrained.return_value = model
    modeling_mod.ESMFold2Model = esmfold2_cls  # type: ignore[attr-defined]

    transformers_mod.models = models_mod  # type: ignore[attr-defined]
    models_mod.esmfold2 = esmfold2_mod  # type: ignore[attr-defined]
    esmfold2_mod.modeling_esmfold2 = modeling_mod  # type: ignore[attr-defined]

    return transformers_mod, model


def _transformers_sys_modules(mod: ModuleType) -> dict[str, ModuleType]:
    models = mod.models  # type: ignore[attr-defined]
    esmfold2 = models.esmfold2
    modeling = esmfold2.modeling_esmfold2
    return {
        "transformers": mod,
        "transformers.models": models,
        "transformers.models.esmfold2": esmfold2,
        "transformers.models.esmfold2.modeling_esmfold2": modeling,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_backend():  # type: ignore[no-untyped-def]
    """Backend with the fake esm + transformers modules wired in."""
    from sheaf.backends.esmfold2 import ESMFold2Backend

    esm_mod = _make_esm_mod()
    transformers_mod, _model = _make_transformers_mod()
    _FakeInputBuilder.result = _make_result()

    backend = ESMFold2Backend(model_name="biohub/ESMFold2", device="cpu")
    mods = {
        **_esm_sys_modules(esm_mod),
        **_transformers_sys_modules(transformers_mod),
    }
    with patch.dict(sys.modules, mods):
        backend.load()
    return backend


def _make_request(
    chains: list[ChainInput] | None = None,
    num_loops: int = 3,
    num_sampling_steps: int = 50,
    num_samples: int = 1,
    seed: int = 0,
    output_format: str = "mmcif",
) -> StructureRequest:
    if chains is None:
        chains = [ChainInput(chain_id="A", sequence="MKTAYIAK")]
    return StructureRequest(
        model_name="esmfold2",
        chains=chains,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_samples=num_samples,
        seed=seed,
        output_format=output_format,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# load() — error cases
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_esm() -> None:
    from sheaf.backends.esmfold2 import ESMFold2Backend

    backend = ESMFold2Backend()
    mods_without = {k: v for k, v in sys.modules.items() if not k.startswith("esm")}
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name == "esm" or name.startswith("esm."):
            raise ModuleNotFoundError("No module named 'esm'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[protein\\]"),
    ):
        backend.load()


def test_load_rejects_forge_only_model() -> None:
    from sheaf.backends.esmfold2 import ESMFold2Backend

    backend = ESMFold2Backend(model_name="esmfold2-fast-2026-05")
    with pytest.raises(NotImplementedError, match="Forge"):
        backend.load()


# ---------------------------------------------------------------------------
# load() — happy path
# ---------------------------------------------------------------------------


def test_load_passes_model_name_and_device() -> None:
    from sheaf.backends.esmfold2 import ESMFold2Backend

    esm_mod = _make_esm_mod()
    transformers_mod, model = _make_transformers_mod()
    backend = ESMFold2Backend(model_name="biohub/ESMFold2", device="cuda:1")
    mods = {
        **_esm_sys_modules(esm_mod),
        **_transformers_sys_modules(transformers_mod),
    }
    with patch.dict(sys.modules, mods):
        backend.load()

    modeling = transformers_mod.models.esmfold2.modeling_esmfold2  # type: ignore[attr-defined]
    modeling.ESMFold2Model.from_pretrained.assert_called_once_with(  # type: ignore[attr-defined]
        "biohub/ESMFold2"
    )
    model.to.assert_called_once_with("cuda:1")


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.molecular import MolecularRequest

    req = MolecularRequest(model_name="x", sequences=["MKT"])
    with pytest.raises(TypeError, match="StructureRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — chain assembly & scaling parameter wiring
# ---------------------------------------------------------------------------


def test_predict_builds_protein_input_per_chain(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result()
    chains = [
        ChainInput(chain_id="A", sequence="MKTAYIAK"),
        ChainInput(chain_id="B", sequence="ACDEFG"),
    ]
    loaded_backend.predict(_make_request(chains=chains))

    kwargs = _FakeInputBuilder.last_call_kwargs
    assert kwargs is not None
    assert kwargs["num_loops"] == 3  # default
    assert kwargs["num_sampling_steps"] == 50
    assert kwargs["num_diffusion_samples"] == 1
    assert kwargs["seed"] == 0


def test_predict_threads_inference_params(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(sample_scores=[0.71, 0.55, 0.83, 0.61])
    loaded_backend.predict(
        _make_request(
            num_loops=5,
            num_sampling_steps=100,
            num_samples=4,
            seed=42,
        )
    )

    kwargs = _FakeInputBuilder.last_call_kwargs
    assert kwargs == {
        "num_loops": 5,
        "num_sampling_steps": 100,
        "num_diffusion_samples": 4,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_structure_response_mmcif(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(mmcif="MY-CIF")
    resp = loaded_backend.predict(_make_request(output_format="mmcif"))

    assert isinstance(resp, StructureResponse)
    assert resp.structure_format == "mmcif"
    assert resp.structure == "MY-CIF"


def test_predict_returns_structure_response_pdb(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(pdb="MY-PDB")
    resp = loaded_backend.predict(_make_request(output_format="pdb"))

    assert resp.structure_format == "pdb"
    assert resp.structure == "MY-PDB"


def test_predict_copies_confidence_metrics(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(
        plddt=[60.0, 70.5, 80.0, 90.5],
        ptm=0.88,
        iptm=0.55,
        pae=[[0.0, 1.2], [1.2, 0.0]],
    )
    resp = loaded_backend.predict(_make_request())

    assert resp.plddt == [60.0, 70.5, 80.0, 90.5]
    assert resp.ptm == pytest.approx(0.88)
    assert resp.iptm == pytest.approx(0.55)
    assert resp.pae == [[0.0, 1.2], [1.2, 0.0]]


def test_predict_skips_optional_confidence_when_missing(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(ptm=None, iptm=None, pae=None)
    resp = loaded_backend.predict(_make_request())

    assert resp.ptm is None
    assert resp.iptm is None
    assert resp.pae is None


def test_predict_omits_sample_scores_when_num_samples_is_one(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(sample_scores=[0.9])
    resp = loaded_backend.predict(_make_request(num_samples=1))

    # num_samples=1 → no sample_scores in response, regardless of upstream.
    assert resp.sample_scores is None


def test_predict_includes_sample_scores_when_num_samples_gt_one(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result(sample_scores=[0.7, 0.9, 0.6])
    resp = loaded_backend.predict(_make_request(num_samples=3))

    assert resp.sample_scores == [0.7, 0.9, 0.6]


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_each_request(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    _FakeInputBuilder.result = _make_result()
    reqs = [_make_request(), _make_request(seed=7)]
    responses = loaded_backend.batch_predict(reqs)

    assert len(responses) == 2
    assert all(isinstance(r, StructureResponse) for r in responses)
