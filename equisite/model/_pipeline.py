"""Public inference pipeline API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .._device import resolve_device
from .._inference_runner import run_single_inference
from .._types import BinaryPredictionRow
from ._model import EquiSite
from ._result import BinaryPredictionResult, PredictionResult


class EquiSitePipeline:
    """High-level prediction pipeline around an EquiSite model instance."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | int | torch.device | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize a pipeline with an existing model."""
        self.device = (
            resolve_device(device) if device is not None else next(model.parameters()).device
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.model_path = Path(model_path) if model_path is not None else None

    @classmethod
    def from_pretrained(
        cls,
        *,
        binding_type: str = "DNA",
        model_path: str | Path | None = None,
        device: str | int | torch.device | None = "0",
        model_kwargs: dict[str, Any] | None = None,
    ) -> EquiSitePipeline:
        """Build a pipeline by loading pretrained model weights."""
        model, resolved_device, resolved_model_path = EquiSite.from_pretrained(
            binding_type=binding_type,
            model_path=model_path,
            device=device,
            model_kwargs=model_kwargs,
        )
        return cls(model=model, device=resolved_device, model_path=resolved_model_path)

    def predict_proba(
        self,
        pdb_path: str | Path,
        *,
        sequence: str | None = None,
    ) -> PredictionResult:
        """Predict per-residue binding probabilities for a single protein."""
        rows = run_single_inference(pdb_path, self.model, self.device, sequence=sequence)
        return PredictionResult(rows=rows)

    def predict(
        self,
        pdb_path: str | Path,
        *,
        threshold: float = 0.5,
        sequence: str | None = None,
    ) -> BinaryPredictionResult:
        """Predict per-residue binding labels at ``threshold``."""
        probability_rows = self.predict_proba(pdb_path, sequence=sequence)
        binary_rows: list[BinaryPredictionRow] = []
        for row in probability_rows:
            binary_row: BinaryPredictionRow = {
                **row,
                "is_binding": row["binding_probability"] >= threshold,
            }
            binary_rows.append(binary_row)
        return BinaryPredictionResult(rows=binary_rows)

    def eval(self) -> EquiSitePipeline:
        """Set model to eval mode and return the pipeline."""
        self.model.eval()
        return self
