"""Backward-compatible predictor wrapper around the public model pipeline API."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ._types import BinaryPredictionRow, PredictionRow
from .model import EquiSitePipeline


class EquiSitePredictor:
    """Compatibility wrapper mirroring the previous predictor interface."""

    def __init__(
        self, model: nn.Module, device: torch.device, model_path: Path | None = None
    ) -> None:
        """Build a predictor from a preloaded model and target device."""
        self._pipeline = EquiSitePipeline(model=model, device=device, model_path=model_path)

    @property
    def model(self) -> nn.Module:
        """Expose the underlying model instance."""
        return self._pipeline.model

    @property
    def device(self) -> torch.device:
        """Expose the predictor device."""
        return self._pipeline.device

    @property
    def model_path(self) -> Path | None:
        """Expose the checkpoint path when known."""
        return self._pipeline.model_path

    @classmethod
    def from_pretrained(
        cls,
        *,
        binding_type: str = "DNA",
        model_path: str | Path | None = None,
        device: str | int | torch.device | None = "0",
        model_kwargs: dict[str, object] | None = None,
    ) -> EquiSitePredictor:
        """Instantiate a predictor from pretrained checkpoint weights."""
        pipeline = EquiSitePipeline.from_pretrained(
            binding_type=binding_type,
            model_path=model_path,
            device=device,
            model_kwargs=model_kwargs,
        )
        return cls(model=pipeline.model, device=pipeline.device, model_path=pipeline.model_path)

    def predict_proba(
        self,
        pdb_path: str | Path,
        *,
        sequence: str | None = None,
    ) -> list[PredictionRow]:
        """Return per-residue binding probabilities for a single PDB."""
        return self._pipeline.predict_proba(pdb_path, sequence=sequence).to_list()

    def predict(
        self,
        pdb_path: str | Path,
        *,
        threshold: float = 0.5,
        sequence: str | None = None,
    ) -> list[BinaryPredictionRow]:
        """Return per-residue binary predictions at a probability threshold."""
        probabilities = self.predict_proba(pdb_path, sequence=sequence)
        return [
            {
                **row,
                "is_binding": row["binding_probability"] >= threshold,
            }
            for row in probabilities
        ]
