"""Model loading helpers for EquiSite inference."""

from __future__ import annotations

from pathlib import Path

import torch

from model.equisite_t3_pro import EquiSite


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
