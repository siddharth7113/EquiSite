"""Unit tests for sequence extraction from HDF5 pipeline input."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_extract_sequence_compacts_atom_level_records(tmp_path: Path) -> None:
    """Convert atom-level residue records into residue-level sequence."""
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from equisite.preprocessing._hdf5_io import extract_sequence

    file_path = tmp_path / "protein.hdf5"
    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset(
            "atom_residue_names",
            data=np.array([b"GLY", b"GLY", b"ALA", b"UNK", b"UNK"]),
        )
        h5_file.create_dataset("atom_amino_id", data=np.array([0, 0, 1, 2, 2]))

    sequence, residue_names = extract_sequence(file_path)
    assert sequence == "GAX"
    assert residue_names == ["GLY", "ALA", "UNK"]
