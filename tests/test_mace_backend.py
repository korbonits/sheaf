"""Tests for MACEBackend — fully mocked, no mace-torch or ase required.

Covers:
  - load() raises ImportError when mace-torch is absent
  - load() calls mace_mp with correct model/device/dtype/dispersion args
  - load() stores Atoms class for test injectability
  - predict() rejects non-MaterialsRequest inputs
  - predict() returns MaterialsResponse with correct structure
  - predict() energy value matches FakeAtoms
  - predict() forces_b64 decodes to shape (N, 3)
  - predict() forces skipped when compute_forces=False
  - predict() stress computed when compute_stress=True
  - predict() stress is None when compute_stress=False
  - predict() n_atoms matches len(atomic_numbers)
  - predict() passes correct numbers and positions to Atoms constructor
  - predict() passes cell and pbc for periodic systems
  - batch_predict() runs each request independently
"""

from __future__ import annotations

import base64
import builtins
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sheaf.api.materials import MaterialsRequest, MaterialsResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_ATOMS = 3
_FAKE_ENERGY = -13.6  # eV
# CO2-like: C at origin, two O atoms
_ATOMIC_NUMBERS = [6, 8, 8]
_POSITIONS = np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]], dtype=np.float32
)

# ---------------------------------------------------------------------------
# FakeAtoms — ASE Atoms stand-in; captures constructor args
# ---------------------------------------------------------------------------


class FakeAtoms:
    """Minimal ASE Atoms mock. Records constructor kwargs; returns fixed outputs."""

    last_call: dict = {}

    def __init__(
        self,
        numbers: list[int],
        positions: np.ndarray,
        pbc: bool | list[bool] = False,
        cell: list[list[float]] | None = None,
    ) -> None:
        FakeAtoms.last_call = {
            "numbers": list(numbers),
            "positions": np.array(positions),
            "pbc": pbc,
            "cell": cell,
        }
        self._numbers = numbers
        self.calc = None

    def get_potential_energy(self) -> float:
        return _FAKE_ENERGY

    def get_forces(self) -> np.ndarray:
        return np.zeros((len(self._numbers), 3), dtype=np.float32)

    def get_stress(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake mace + ase module factories
# ---------------------------------------------------------------------------


def _make_mace_mods() -> tuple[ModuleType, MagicMock]:
    """Return (mace module tree, fake_calc)."""
    fake_calc = MagicMock()
    mace_mod = ModuleType("mace")
    calc_mod = ModuleType("mace.calculators")
    calc_mod.mace_mp = MagicMock(return_value=fake_calc)  # type: ignore[attr-defined]
    mace_mod.calculators = calc_mod  # type: ignore[attr-defined]
    return mace_mod, fake_calc


def _make_ase_mod() -> ModuleType:
    ase_mod = ModuleType("ase")
    ase_mod.Atoms = FakeAtoms  # type: ignore[attr-defined]
    return ase_mod


def _sys_mods(mace_mod: ModuleType, ase_mod: ModuleType) -> dict[str, ModuleType]:
    return {
        "mace": mace_mod,
        "mace.calculators": mace_mod.calculators,  # type: ignore[attr-defined]
        "ase": ase_mod,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enc(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _positions_b64(positions: np.ndarray = _POSITIONS) -> str:
    return _enc(positions)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mace_mods() -> tuple[ModuleType, MagicMock]:
    return _make_mace_mods()


@pytest.fixture
def ase_mod() -> ModuleType:
    return _make_ase_mod()


@pytest.fixture
def loaded_backend(mace_mods: tuple[ModuleType, MagicMock], ase_mod: ModuleType):  # type: ignore[no-untyped-def]
    from sheaf.backends.mace import MACEBackend

    mace_mod, fake_calc = mace_mods
    backend = MACEBackend(model="medium", device="cpu")
    with patch.dict(sys.modules, _sys_mods(mace_mod, ase_mod)):
        backend.load()
    # Inject FakeAtoms so predict() builds FakeAtoms (not real ase.Atoms)
    backend._Atoms = FakeAtoms
    backend._calc = fake_calc
    return backend


# ---------------------------------------------------------------------------
# Request factory
# ---------------------------------------------------------------------------


def _make_request(
    atomic_numbers: list[int] | None = None,
    positions: np.ndarray | None = None,
    cell: list[list[float]] | None = None,
    pbc: bool | list[bool] = False,
    compute_forces: bool = True,
    compute_stress: bool = False,
) -> MaterialsRequest:
    if atomic_numbers is None:
        atomic_numbers = _ATOMIC_NUMBERS
    if positions is None:
        positions = _POSITIONS
    return MaterialsRequest(
        model_name="mace",
        atomic_numbers=atomic_numbers,
        positions_b64=_positions_b64(positions),
        cell=cell,
        pbc=pbc,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_mace() -> None:
    from sheaf.backends.mace import MACEBackend

    backend = MACEBackend()
    mods_without = {
        k: v for k, v in sys.modules.items() if not k.startswith("mace") and k != "ase"
    }
    _real_import = builtins.__import__

    def _raise(name: str, *a: object, **kw: object) -> object:
        if name in ("mace", "mace.calculators", "ase"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return _real_import(name, *a, **kw)

    with (
        patch.dict(sys.modules, mods_without, clear=True),
        patch("builtins.__import__", side_effect=_raise),
        pytest.raises(ImportError, match="sheaf-serve\\[materials\\]"),
    ):
        backend.load()


def test_load_calls_mace_mp_with_args(
    mace_mods: tuple[ModuleType, MagicMock], ase_mod: ModuleType
) -> None:
    from sheaf.backends.mace import MACEBackend

    mace_mod, _ = mace_mods
    backend = MACEBackend(
        model="large", device="cuda", default_dtype="float64", dispersion=True
    )
    with patch.dict(sys.modules, _sys_mods(mace_mod, ase_mod)):
        backend.load()

    mace_mod.calculators.mace_mp.assert_called_once_with(  # type: ignore[attr-defined]
        model="large",
        device="cuda",
        default_dtype="float64",
        dispersion=True,
    )


def test_load_stores_atoms_cls(
    mace_mods: tuple[ModuleType, MagicMock], ase_mod: ModuleType
) -> None:
    from sheaf.backends.mace import MACEBackend

    mace_mod, _ = mace_mods
    backend = MACEBackend()
    assert backend._Atoms is None
    with patch.dict(sys.modules, _sys_mods(mace_mod, ase_mod)):
        backend.load()
    assert backend._Atoms is FakeAtoms


# ---------------------------------------------------------------------------
# predict() — input validation
# ---------------------------------------------------------------------------


def test_predict_rejects_wrong_type(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    from sheaf.api.genomic import GenomicRequest

    req = GenomicRequest(model_name="x", sequences=["ATCG"])
    with pytest.raises(TypeError, match="MaterialsRequest"):
        loaded_backend.predict(req)


# ---------------------------------------------------------------------------
# predict() — response structure
# ---------------------------------------------------------------------------


def test_predict_returns_materials_response(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    resp = loaded_backend.predict(_make_request())
    assert isinstance(resp, MaterialsResponse)


def test_predict_energy_value(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    resp = loaded_backend.predict(_make_request())
    assert resp.energy == pytest.approx(_FAKE_ENERGY)


def test_predict_n_atoms_matches_input(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    resp = loaded_backend.predict(_make_request())
    assert resp.n_atoms == _N_ATOMS


# ---------------------------------------------------------------------------
# predict() — forces
# ---------------------------------------------------------------------------


def test_predict_forces_shape(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """forces_b64 decodes to (N, 3) float32 array."""
    resp = loaded_backend.predict(_make_request(compute_forces=True))
    assert resp.forces_b64 is not None
    forces = np.frombuffer(base64.b64decode(resp.forces_b64), dtype=np.float32).reshape(
        _N_ATOMS, 3
    )
    assert forces.shape == (_N_ATOMS, 3)


def test_predict_forces_none_when_not_requested(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    resp = loaded_backend.predict(_make_request(compute_forces=False))
    assert resp.forces_b64 is None


# ---------------------------------------------------------------------------
# predict() — stress
# ---------------------------------------------------------------------------


def test_predict_stress_computed_when_requested(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    """stress_b64 decodes to (6,) Voigt float32 array."""
    resp = loaded_backend.predict(_make_request(compute_stress=True))
    assert resp.stress_b64 is not None
    stress = np.frombuffer(base64.b64decode(resp.stress_b64), dtype=np.float32)
    assert stress.shape == (6,)


def test_predict_stress_none_when_not_requested(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    resp = loaded_backend.predict(_make_request(compute_stress=False))
    assert resp.stress_b64 is None


# ---------------------------------------------------------------------------
# predict() — Atoms construction
# ---------------------------------------------------------------------------


def test_predict_passes_correct_numbers_and_positions(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    loaded_backend.predict(_make_request())
    assert FakeAtoms.last_call["numbers"] == _ATOMIC_NUMBERS
    np.testing.assert_array_almost_equal(FakeAtoms.last_call["positions"], _POSITIONS)


def test_predict_periodic_system_passes_cell_and_pbc(
    loaded_backend,  # type: ignore[no-untyped-def]
) -> None:
    """cell and pbc are forwarded to the Atoms constructor."""
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    pbc = [True, True, True]
    loaded_backend.predict(_make_request(cell=cell, pbc=pbc))
    assert FakeAtoms.last_call["cell"] == cell
    assert FakeAtoms.last_call["pbc"] == pbc


# ---------------------------------------------------------------------------
# batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_runs_independently(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    reqs = [_make_request(), _make_request()]
    responses = loaded_backend.batch_predict(reqs)
    assert len(responses) == 2
    assert all(isinstance(r, MaterialsResponse) for r in responses)
    assert all(r.energy == pytest.approx(_FAKE_ENERGY) for r in responses)
