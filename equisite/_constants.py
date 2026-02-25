"""Internal constants for EquiSite bindings."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_CHECKPOINTS = {
    "DNA": PROJECT_ROOT / "checkpoints" / "DNA" / "best_val.pt",
    "RNA": PROJECT_ROOT / "checkpoints" / "RNA" / "best_val.pt",
}

AA3_TO_1 = {
    "GLY": "G",
    "ALA": "A",
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "PHE": "F",
    "PRO": "P",
    "MET": "M",
    "TRP": "W",
    "CYS": "C",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "TYR": "Y",
    "HIS": "H",
    "ASP": "D",
    "GLU": "E",
    "LYS": "K",
    "ARG": "R",
    "UNK": "X",
}
