"""Unit tests for internal geometry helper functions."""

from __future__ import annotations

import pytest


def _load_geometry_dependencies():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from equisite.preprocessing import _geometry

    return np, torch, _geometry


def test_compute_dihedrals_returns_finite_values() -> None:
    """Produce finite torsion values for simple vectors."""
    _np, torch, geometry = _load_geometry_dependencies()
    v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v3 = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    angles = geometry.compute_dihedrals(v1, v2, v3)
    assert angles.shape == (2,)
    assert torch.isfinite(angles).all()


def test_backbone_embeddings_shape() -> None:
    """Generate 6-dimensional backbone embeddings per residue."""
    _np, torch, geometry = _load_geometry_dependencies()
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.5, 0.0]],
            [[2.0, 0.5, 0.1], [2.8, 0.9, 0.2], [3.4, 1.3, 0.4]],
            [[3.9, 1.6, 0.6], [4.5, 2.0, 0.8], [5.0, 2.4, 0.9]],
        ]
    )

    embeddings = geometry.backbone_embeddings(positions)
    assert embeddings.shape == (3, 6)
    assert torch.isfinite(embeddings).all()


def test_get_atom_positions_falls_back_to_ca() -> None:
    """Use CA coordinates when residue misses N/C atoms."""
    np, torch, geometry = _load_geometry_dependencies()
    amino_types = np.array([0, 1])
    atom_names = np.array([b"N", b"CA", b"C", b"CA"])
    atom_amino_id = np.array([0, 0, 0, 1])
    atom_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )

    pos_n, pos_ca, pos_c, *_ = geometry.get_atom_positions(
        amino_types, atom_names, atom_amino_id, atom_pos
    )
    assert torch.allclose(pos_n[1], pos_ca[1])
    assert torch.allclose(pos_c[1], pos_ca[1])
