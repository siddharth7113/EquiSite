"""PyG graph construction for EquiSite inference."""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch_geometric.data import Data

from ._esm import compute_esm_embeddings
from ._geometry import backbone_embeddings, get_atom_positions, side_chain_embeddings


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
