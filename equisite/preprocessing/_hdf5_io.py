"""HDF5 conversion and sequence extraction helpers."""

from __future__ import annotations

from pathlib import Path

import h5py

from .._constants import AA3_TO_1
from .py_periodic_table import PyPeriodicTable
from .py_protein import PyProtein


def pdb_to_hdf5(pdb_path: str | Path, hdf5_path: str | Path) -> None:
    """Convert a cleaned PDB file to the internal HDF5 representation."""
    protein = PyProtein(PyPeriodicTable())
    protein.load_molecular_file(str(pdb_path))
    protein.compute_covalent_bonds()
    protein.compute_hydrogen_bonds()
    protein.save_hdf5(str(hdf5_path))


def extract_sequence(hdf5_path: str | Path) -> tuple[str, list[str]]:
    """Extract one-letter sequence and residue names from an HDF5 protein file."""
    with h5py.File(str(hdf5_path), "r") as h5_file:
        residue_names = [value.decode("utf-8") for value in h5_file["atom_residue_names"][()]]
        amino_ids = h5_file["atom_amino_id"][()]

    sequence: list[str] = []
    residue_name_list: list[str] = []
    previous_amino_id = None
    for residue_name, amino_id in zip(residue_names, amino_ids):
        if amino_id != previous_amino_id:
            sequence.append(AA3_TO_1.get(residue_name, "X"))
            residue_name_list.append(residue_name)
        previous_amino_id = amino_id

    return "".join(sequence), residue_name_list
