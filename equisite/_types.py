"""Internal data contracts for EquiSite inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class PDBResidueRecord:
    """Residue identity fields used to map predictions back to PDB records."""

    chain: str
    residue_index: int
    insertion_code: str
    residue_name: str


class PredictionRow(TypedDict):
    """Per-residue probability output row."""

    residue_index: int
    chain: str
    insertion_code: str
    residue_name: str
    binding_probability: float


class BinaryPredictionRow(PredictionRow):
    """Per-residue binary output row."""

    is_binding: bool
