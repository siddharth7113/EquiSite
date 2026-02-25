"""Public EquiSite model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model.equisite_t3_pro import EquiSite as _LegacyEquiSite

from .._constants import DEFAULT_CHECKPOINTS
from .._device import resolve_device
from .._model_loader import load_checkpoint_weights


class EquiSite(_LegacyEquiSite):
    """Public EquiSite model class for programmatic usage."""

    def __init__(
        self,
        *,
        args: Any = None,
        level: str = "allatom+esm",
        num_blocks: int = 4,
        hidden_channels: int = 128,
        out_channels: int = 1,
        mid_emb: int = 64,
        num_radial: int = 6,
        num_spherical: int = 3,
        cutoff: float = 11.5,
        max_num_neighbors: int = 32,
        int_emb_layers: int = 3,
        out_layers: int = 2,
        num_pos_emb: int = 16,
        dropout: float = 0.25,
        data_augment_eachlayer: bool = False,
        euler_noise: bool = False,
    ) -> None:
        """Initialize an EquiSite model with inference-friendly defaults."""
        super().__init__(
            args=args,
            level=level,
            num_blocks=num_blocks,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            mid_emb=mid_emb,
            num_radial=num_radial,
            num_spherical=num_spherical,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            int_emb_layers=int_emb_layers,
            out_layers=out_layers,
            num_pos_emb=num_pos_emb,
            dropout=dropout,
            data_augment_eachlayer=data_augment_eachlayer,
            euler_noise=euler_noise,
        )

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | int | torch.device | None = None,
    ) -> EquiSite:
        """Load checkpoint weights into this model instance and set eval mode."""
        resolved_device = resolve_device(device)
        load_checkpoint_weights(self, checkpoint_path, resolved_device)
        return self

    @classmethod
    def from_pretrained(
        cls,
        *,
        binding_type: str = "DNA",
        model_path: str | Path | None = None,
        device: str | int | torch.device | None = "0",
        model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[EquiSite, torch.device, Path]:
        """Create a model, load checkpoint weights, and return model with metadata."""
        normalized_binding_type = binding_type.upper()
        if normalized_binding_type not in DEFAULT_CHECKPOINTS:
            raise ValueError(
                f"Unsupported binding_type '{binding_type}'. Expected one of: DNA, RNA."
            )

        resolved_device = resolve_device(device)
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

        effective_model_kwargs = dict(model_kwargs or {})
        model = cls(**effective_model_kwargs)
        model.load_pretrained_weights(checkpoint_path, device=resolved_device)
        return model, resolved_device, checkpoint_path
