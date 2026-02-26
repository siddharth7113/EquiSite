"""PDB cleaning and residue record extraction utilities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa

from .._types import PDBResidueRecord


class _ProteinOnlySelect(Select):
    """Keep only protein residues and canonical alternate atom locations."""

    def accept_residue(self, residue: Any) -> int:
        hetfield, _, _ = residue.id
        if hetfield.strip():
            return 0
        residue_name = residue.get_resname().strip()
        return int(is_aa(residue, standard=True) or residue_name == "UNK")

    def accept_atom(self, atom: Any) -> int:
        altloc = atom.get_altloc().strip()
        if not atom.is_disordered():
            return 1
        return int(altloc in {"", "A", "1"})


def _load_structure(pdb_path: str | Path) -> Any:
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", str(pdb_path))


def iter_protein_residues(pdb_path: str | Path) -> Iterator[PDBResidueRecord]:
    """Yield protein residues from a PDB file in structural traversal order."""
    structure = _load_structure(pdb_path)
    for model in structure:
        for chain in model:
            chain_id = chain.id.strip() or "?"
            for residue in chain:
                hetfield, residue_sequence, insertion_code = residue.id
                if hetfield.strip():
                    continue
                residue_name = residue.get_resname().strip()
                if not (is_aa(residue, standard=True) or residue_name == "UNK"):
                    continue
                yield PDBResidueRecord(
                    chain=chain_id,
                    residue_index=int(residue_sequence),
                    insertion_code=str(insertion_code).strip(),
                    residue_name=residue_name,
                )


def remove_hetero_atoms(src: str | Path, dst: str | Path) -> None:
    """Write a protein-only PDB file."""
    structure = _load_structure(src)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst), select=_ProteinOnlySelect())


def extract_pdb_residue_numbers(pdb_path: str | Path) -> list[dict[str, int | str]]:
    """Extract residue numbering fields from a cleaned PDB file."""
    return [
        {
            "chain": residue.chain,
            "resSeq": residue.residue_index,
            "iCode": residue.insertion_code,
            "resName": residue.residue_name,
        }
        for residue in iter_protein_residues(pdb_path)
    ]
