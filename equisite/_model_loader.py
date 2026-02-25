"""Model loading helpers for EquiSite inference."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load checkpoint weights into an initialized model and set eval mode."""
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
