"""API contract for materials / interatomic potential models (MACE-MP, etc.)."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from sheaf.api.base import BaseRequest, BaseResponse, ModelType


class MaterialsRequest(BaseRequest):
    """Request contract for atomistic energy/force/stress prediction.

    Describes a single atomic structure: a set of atoms at given positions,
    optionally in a periodic simulation cell. The model predicts the potential
    energy surface and its derivatives.

    Args:
        atomic_numbers: Atomic numbers (Z) for each atom. Length N.
        positions_b64: Base64-encoded float32 array of shape (N, 3) giving
            Cartesian coordinates in Angstroms.
        cell: 3x3 lattice vectors in Angstroms for periodic boundary
            conditions. Required when ``pbc`` is True.
        pbc: Periodic boundary conditions. ``False`` (default) for isolated
            molecules/clusters; ``True`` or ``[True, True, True]`` for
            bulk crystals; ``[True, True, False]`` for slabs.
        compute_forces: If True (default), return forces in eV/Å.
        compute_stress: If True, return the stress tensor in eV/Å³ (Voigt).
            Only meaningful for periodic systems (``pbc=True``).
    """

    model_type: Literal[ModelType.MATERIALS] = ModelType.MATERIALS

    atomic_numbers: list[int]
    positions_b64: str  # base64 float32, shape (N, 3) in Angstroms
    cell: list[list[float]] | None = None  # 3×3 lattice vectors in Angstroms
    pbc: bool | list[bool] = False
    compute_forces: bool = True
    compute_stress: bool = False

    @model_validator(mode="after")
    def _validate_structure(self) -> MaterialsRequest:
        if len(self.atomic_numbers) == 0:
            raise ValueError("atomic_numbers must not be empty")
        if self.cell is not None:
            if len(self.cell) != 3 or any(len(row) != 3 for row in self.cell):
                raise ValueError("cell must be a 3×3 matrix")
        if isinstance(self.pbc, list) and len(self.pbc) != 3:
            raise ValueError("pbc list must have exactly 3 elements")
        return self


class MaterialsResponse(BaseResponse):
    """Response contract for atomistic energy/force/stress prediction."""

    model_type: Literal[ModelType.MATERIALS] = ModelType.MATERIALS

    # Potential energy in eV
    energy: float

    # Forces in eV/Å — base64 float32, shape (N, 3). None if compute_forces=False.
    forces_b64: str | None = None

    # Stress tensor (Voigt notation) in eV/Å³ — base64 float32, shape (6,).
    # None if compute_stress=False.
    stress_b64: str | None = None

    # Number of atoms in the structure
    n_atoms: int
