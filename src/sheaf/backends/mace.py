"""MACE-MP backend for atomistic energy, force, and stress prediction.

Requires: pip install "sheaf-serve[materials]"
Library:  mace-torch (https://github.com/ACEsuit/mace)

MACE-MP-0 is a universal interatomic potential trained on the MPTrj dataset
(~1.5M DFT calculations across ~90 elements from the Materials Project).
It can predict energy, forces, and stress for any inorganic material or
organic molecule without fine-tuning.

Supported model sizes:
  "small"  — ~2M params, fast (~1ms/atom on CPU)
  "medium" — ~6M params, recommended balance (default)
  "large"  — ~17M params, highest accuracy

Outputs (all in eV / Angstrom units, matching ASE conventions):
  energy  — potential energy in eV
  forces  — Cartesian forces in eV/Å, shape (N, 3)
  stress  — Voigt stress tensor in eV/Å³, shape (6,)  [periodic only]

ASE ``Atoms`` is stored at load() time so tests can inject a mock class
without ase installed in the test environment.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from sheaf.api.base import BaseRequest, BaseResponse, ModelType
from sheaf.api.materials import MaterialsRequest, MaterialsResponse
from sheaf.backends.base import ModelBackend
from sheaf.registry import register_backend


@register_backend("mace")
class MACEBackend(ModelBackend):
    """ModelBackend for MACE-MP-0 universal interatomic potential.

    Wraps the MACE calculator as an ASE-compatible calculator. Each predict()
    call constructs an ASE ``Atoms`` object, attaches the calculator, and
    queries energy, forces, and optionally stress.

    Args:
        model: MACE-MP model size — "small", "medium" (default), or "large".
        device: "cpu", "cuda", or "mps"
        default_dtype: "float32" (default) or "float64"
        dispersion: If True, add D3 dispersion correction (requires torch-dftd3).
    """

    def __init__(
        self,
        model: str = "medium",
        device: str = "cpu",
        default_dtype: str = "float32",
        dispersion: bool = False,
    ) -> None:
        self._model = model
        self._device = device
        self._default_dtype = default_dtype
        self._dispersion = dispersion
        self._calc: Any = None
        self._Atoms: Any = None  # ase.Atoms, stored at load() for test injectability

    @property
    def model_type(self) -> str:
        return ModelType.MATERIALS

    def load(self) -> None:
        try:
            from ase import Atoms  # ty: ignore[unresolved-import]
            from mace.calculators import mace_mp  # ty: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(
                "mace-torch is required for the MACE backend. "
                "Install it with: pip install 'sheaf-serve[materials]'"
            ) from e
        self._calc = mace_mp(
            model=self._model,
            device=self._device,
            default_dtype=self._default_dtype,
            dispersion=self._dispersion,
        )
        self._Atoms = Atoms

    def predict(self, request: BaseRequest) -> BaseResponse:
        if not isinstance(request, MaterialsRequest):
            raise TypeError(f"Expected MaterialsRequest, got {type(request)}")
        return self._run(request)

    def batch_predict(self, requests: list[BaseRequest]) -> list[BaseResponse]:
        return [self.predict(r) for r in requests]

    def _run(self, request: MaterialsRequest) -> MaterialsResponse:
        if self._calc is None or self._Atoms is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        n = len(request.atomic_numbers)
        positions = np.frombuffer(
            base64.b64decode(request.positions_b64), dtype=np.float32
        ).reshape(n, 3)

        atoms_kwargs: dict[str, Any] = dict(
            numbers=request.atomic_numbers,
            positions=positions,
            pbc=request.pbc,
        )
        if request.cell is not None:
            atoms_kwargs["cell"] = request.cell

        atoms = self._Atoms(**atoms_kwargs)
        atoms.calc = self._calc

        energy = float(atoms.get_potential_energy())

        forces_b64 = None
        if request.compute_forces:
            forces = np.array(atoms.get_forces(), dtype=np.float32)
            forces_b64 = base64.b64encode(forces.tobytes()).decode()

        stress_b64 = None
        if request.compute_stress:
            stress = np.array(atoms.get_stress(), dtype=np.float32)
            stress_b64 = base64.b64encode(stress.tobytes()).decode()

        return MaterialsResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            energy=energy,
            forces_b64=forces_b64,
            stress_b64=stress_b64,
            n_atoms=n,
        )
