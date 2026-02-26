"""Helpers for pretrained model resolution."""

from __future__ import annotations

from pathlib import Path

from .._constants import DEFAULT_CHECKPOINTS


def resolve_checkpoint_path(binding_type: str, model_path: str | Path | None) -> Path:
    """Resolve a checkpoint path for a binding type, validating existence."""
    normalized_binding_type = binding_type.upper()
    if normalized_binding_type not in DEFAULT_CHECKPOINTS:
        raise ValueError(f"Unsupported binding_type '{binding_type}'. Expected one of: DNA, RNA.")

    checkpoint_path = (
        Path(model_path)
        if model_path is not None
        else Path(DEFAULT_CHECKPOINTS[normalized_binding_type])
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Download it from the Zenodo release and place it in the expected location."
        )
    return checkpoint_path
