"""Internal EquiSite inference pipeline primitives."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from torch_geometric.data import Data

from dataset.utils.py_periodic_table import PyPeriodicTable
from dataset.utils.py_protein import PyProtein
from model.equisite_t3_pro import EquiSite

from ._esm import compute_esm_embeddings
from ._geometry import backbone_embeddings, get_atom_positions, side_chain_embeddings

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


@dataclass(frozen=True)
class PDBResidueRecord:
    """Residue identity fields used to map model outputs to PDB indices."""

    chain: str
    residue_index: int
    insertion_code: str
    residue_name: str


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


def _iter_protein_residues(pdb_path: str | Path) -> Iterator[PDBResidueRecord]:
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
    """Extract residue numbering fields from a cleaned PDB."""
    return [
        {
            "chain": residue.chain,
            "resSeq": residue.residue_index,
            "iCode": residue.insertion_code,
            "resName": residue.residue_name,
        }
        for residue in _iter_protein_residues(pdb_path)
    ]


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


def build_graph(hdf5_path: str | Path, sequence: str, device: torch.device) -> Data:
    """Build a PyG graph object from HDF5 structure data and sequence."""
    with h5py.File(str(hdf5_path), "r") as h5_file:
        amino_types = h5_file["amino_types"][()]
        amino_types[amino_types == -1] = 25

        atom_amino_ids = h5_file["atom_amino_id"][()]
        atom_names = h5_file["atom_names"][()]
        atom_positions = h5_file["atom_pos"][()][0]

    graph_data = Data()
    graph_data.esm_emb = compute_esm_embeddings(sequence, device)

    (
        pos_n,
        pos_ca,
        pos_c,
        pos_cb,
        pos_g,
        pos_d,
        pos_e,
        pos_z,
        pos_h,
    ) = get_atom_positions(amino_types, atom_names, atom_amino_ids, atom_positions)

    graph_data.side_chain_embs = side_chain_embeddings(
        pos_n,
        pos_ca,
        pos_c,
        pos_cb,
        pos_g,
        pos_d,
        pos_e,
        pos_z,
        pos_h,
    )
    graph_data.side_chain_embs[torch.isnan(graph_data.side_chain_embs)] = 0

    graph_data.bb_embs = backbone_embeddings(
        torch.cat((pos_n.unsqueeze(1), pos_ca.unsqueeze(1), pos_c.unsqueeze(1)), 1)
    )
    graph_data.bb_embs[torch.isnan(graph_data.bb_embs)] = 0

    graph_data.x = torch.tensor(amino_types).unsqueeze(1)
    graph_data.coords_ca = pos_ca
    graph_data.coords_n = pos_n
    graph_data.coords_c = pos_c
    return graph_data


def load_model(model_path: str | Path, device: torch.device) -> EquiSite:
    """Instantiate an EquiSite model and load checkpoint weights."""
    checkpoint = torch.load(str(model_path), map_location=device)
    model = EquiSite(
        num_blocks=4,
        hidden_channels=128,
        out_channels=1,
        cutoff=11.5,
        dropout=0.25,
        level="allatom+esm",
        args=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_single_inference(
    pdb_path: str | Path,
    model: EquiSite,
    device: torch.device,
    *,
    sequence: str | None = None,
) -> list[dict[str, int | str | float]]:
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

        results: list[dict[str, int | str | float]] = []
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
