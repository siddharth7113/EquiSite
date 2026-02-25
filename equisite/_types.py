"""Internal data contracts for EquiSite inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PDBResidueRecord:
    """Residue identity fields used to map predictions back to PDB records."""

    chain: str
    residue_index: int
    insertion_code: str
    residue_name: str


PredictionRow = dict[str, int | str | float]
