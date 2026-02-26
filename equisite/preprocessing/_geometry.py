"""Internal geometry feature helpers for EquiSite inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize a tensor and replace NaNs with zeros."""
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def compute_dihedrals(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
    """Compute torsion angles from three edge vectors."""
    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)
    a = (n1 * n2).sum(dim=-1)
    b = torch.nan_to_num((torch.cross(n1, n2, dim=-1) * v2).sum(dim=-1) / v2.norm(dim=1))
    return torch.nan_to_num(torch.atan2(b, a))


def get_atom_positions(
    amino_types: np.ndarray,
    atom_names: np.ndarray,
    atom_amino_id: np.ndarray,
    atom_pos: np.ndarray,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Collect residue-wise coordinates for backbone and side-chain anchor atoms."""
    mask_n = np.char.equal(atom_names, b"N")
    mask_ca = np.char.equal(atom_names, b"CA")
    mask_c = np.char.equal(atom_names, b"C")
    mask_cb = np.char.equal(atom_names, b"CB")
    mask_g = (
        np.char.equal(atom_names, b"CG")
        | np.char.equal(atom_names, b"SG")
        | np.char.equal(atom_names, b"OG")
        | np.char.equal(atom_names, b"CG1")
        | np.char.equal(atom_names, b"OG1")
    )
    mask_d = (
        np.char.equal(atom_names, b"CD")
        | np.char.equal(atom_names, b"SD")
        | np.char.equal(atom_names, b"CD1")
        | np.char.equal(atom_names, b"OD1")
        | np.char.equal(atom_names, b"ND1")
    )
    mask_e = (
        np.char.equal(atom_names, b"CE")
        | np.char.equal(atom_names, b"NE")
        | np.char.equal(atom_names, b"OE1")
    )
    mask_z = np.char.equal(atom_names, b"CZ") | np.char.equal(atom_names, b"NZ")
    mask_h = np.char.equal(atom_names, b"NH1")

    pos_n = np.full((len(amino_types), 3), np.nan)
    pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
    pos_n = torch.FloatTensor(pos_n)

    pos_ca = np.full((len(amino_types), 3), np.nan)
    pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
    pos_ca = torch.FloatTensor(pos_ca)

    pos_c = np.full((len(amino_types), 3), np.nan)
    pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
    pos_c = torch.FloatTensor(pos_c)

    pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
    pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

    pos_cb = np.full((len(amino_types), 3), np.nan)
    pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
    pos_cb = torch.FloatTensor(pos_cb)

    pos_g = np.full((len(amino_types), 3), np.nan)
    pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
    pos_g = torch.FloatTensor(pos_g)

    pos_d = np.full((len(amino_types), 3), np.nan)
    pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
    pos_d = torch.FloatTensor(pos_d)

    pos_e = np.full((len(amino_types), 3), np.nan)
    pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
    pos_e = torch.FloatTensor(pos_e)

    pos_z = np.full((len(amino_types), 3), np.nan)
    pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
    pos_z = torch.FloatTensor(pos_z)

    pos_h = np.full((len(amino_types), 3), np.nan)
    pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
    pos_h = torch.FloatTensor(pos_h)

    return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


def side_chain_embeddings(
    pos_n: torch.Tensor,
    pos_ca: torch.Tensor,
    pos_c: torch.Tensor,
    pos_cb: torch.Tensor,
    pos_g: torch.Tensor,
    pos_d: torch.Tensor,
    pos_e: torch.Tensor,
    pos_z: torch.Tensor,
    pos_h: torch.Tensor,
) -> torch.Tensor:
    """Compute side-chain torsion-angle embeddings."""
    del pos_h  # Unused but kept for parity with existing pipeline inputs.

    v1, v2, v3, v4, v5, v6 = (
        pos_ca - pos_n,
        pos_cb - pos_ca,
        pos_g - pos_cb,
        pos_d - pos_g,
        pos_e - pos_d,
        pos_z - pos_e,
    )

    angle1 = torch.unsqueeze(compute_dihedrals(v1, v2, v3), 1)
    angle2 = torch.unsqueeze(compute_dihedrals(v2, v3, v4), 1)
    angle3 = torch.unsqueeze(compute_dihedrals(v3, v4, v5), 1)
    angle4 = torch.unsqueeze(compute_dihedrals(v4, v5, v6), 1)

    angles = torch.cat((angle1, angle2, angle3, angle4), 1)
    return torch.cat((torch.sin(angles), torch.cos(angles)), 1)


def backbone_embeddings(backbone_positions: torch.Tensor) -> torch.Tensor:
    """Compute backbone torsion-angle embeddings for N, CA and C coordinates."""
    coordinates = torch.reshape(backbone_positions, [3 * backbone_positions.shape[0], 3])
    deltas = coordinates[1:] - coordinates[:-1]
    unit_vectors = _normalize(deltas, dim=-1)
    u0 = unit_vectors[:-2]
    u1 = unit_vectors[1:-1]
    u2 = unit_vectors[2:]

    angles = compute_dihedrals(u0, u1, u2)
    angles = F.pad(angles, [1, 2])
    angles = torch.reshape(angles, [-1, 3])
    return torch.cat([torch.cos(angles), torch.sin(angles)], 1)
