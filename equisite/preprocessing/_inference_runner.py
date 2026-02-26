"""Single-file inference orchestration for EquiSite."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .._types import PredictionRow
from ._graph_builder import build_graph
from ._hdf5_io import extract_sequence, pdb_to_hdf5
from ._pdb_cleaning import extract_pdb_residue_numbers, remove_hetero_atoms


def run_single_inference(
    pdb_path: str | Path,
    model: nn.Module,
    device: torch.device,
    *,
    sequence: str | None = None,
) -> list[PredictionRow]:
    """Run EquiSite on a single PDB file and return per-residue probabilities."""
    pdb_path = Path(pdb_path)
    temporary_directory = pdb_path.parent / ".equisite_tmp"
    temporary_directory.mkdir(exist_ok=True)

    file_stem = pdb_path.stem
    cleaned_pdb_path = temporary_directory / f"{file_stem}_clean.pdb"
    hdf5_path = temporary_directory / f"{file_stem}.hdf5"

    try:
        remove_hetero_atoms(pdb_path, cleaned_pdb_path)
        pdb_residues = extract_pdb_residue_numbers(cleaned_pdb_path)
        pdb_to_hdf5(cleaned_pdb_path, hdf5_path)

        extracted_sequence, residue_names = extract_sequence(hdf5_path)
        if len(pdb_residues) != len(residue_names):
            raise ValueError(
                "Residue mapping mismatch between Bio.PDB parsing and HDF5 conversion: "
                f"{len(pdb_residues)} parsed residues vs {len(residue_names)} HDF5 residues."
            )

        sequence_to_use = sequence if sequence is not None else extracted_sequence
        if len(sequence_to_use) != len(residue_names):
            raise ValueError(
                "Sequence length mismatch: "
                f"{len(sequence_to_use)} sequence residues vs "
                f"{len(residue_names)} structural residues."
            )

        graph_data = build_graph(hdf5_path, sequence_to_use, device).to(device)

        with torch.no_grad():
            prediction, _, _ = model(graph_data)

        scores = prediction.squeeze().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        if len(scores) != len(residue_names):
            raise ValueError(
                "Prediction length mismatch: "
                f"{len(scores)} scores vs {len(residue_names)} residue names."
            )

        results: list[PredictionRow] = []
        for index, probability in enumerate(scores):
            residue_name = residue_names[index]
            pdb_residue = pdb_residues[index]
            results.append(
                {
                    "residue_index": pdb_residue["resSeq"],
                    "chain": pdb_residue["chain"],
                    "insertion_code": pdb_residue["iCode"],
                    "residue_name": residue_name,
                    "binding_probability": round(float(probability), 6),
                }
            )
        return results
    finally:
        for file_path in (cleaned_pdb_path, hdf5_path):
            if file_path.exists():
                file_path.unlink()
        if temporary_directory.exists() and not any(temporary_directory.iterdir()):
            temporary_directory.rmdir()
