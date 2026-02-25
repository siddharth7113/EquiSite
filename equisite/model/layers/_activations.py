"""Activation functions for EquiSite model components."""

from __future__ import annotations

import torch


def swish(x: torch.Tensor) -> torch.Tensor:
    """Apply the swish non-linearity."""
    return x * torch.sigmoid(x)
