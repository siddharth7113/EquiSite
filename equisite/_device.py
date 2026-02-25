"""Device resolution helpers for EquiSite inference."""

from __future__ import annotations

import torch


def resolve_device(device: str | int | torch.device | None) -> torch.device:
    """Resolve user device input into a ``torch.device`` value."""
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    value = device.strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value if torch.cuda.is_available() else "cpu")

    index = int(value)
    return torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu")
