"""Public inference API for EquiSite."""

from __future__ import annotations

from pathlib import Path

import torch

from model.equisite_t3_pro import EquiSite

from ._pipeline import DEFAULT_CHECKPOINTS, load_model, run_single_inference


def _resolve_device(device: str | int | torch.device | None) -> torch.device:
    """Resolve a user-provided device value to a ``torch.device``."""
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


class EquiSitePredictor:
    """Sklearn-style inference wrapper around the EquiSite model."""

    def __init__(
        self, model: EquiSite, device: torch.device, model_path: Path | None = None
    ) -> None:
        """Build a predictor from a preloaded model and target device."""
        self.model = model
        self.device = device
        self.model_path = model_path

    @classmethod
    def from_pretrained(
        cls,
        *,
        binding_type: str = "DNA",
        model_path: str | Path | None = None,
        device: str | int | torch.device | None = "0",
    ) -> EquiSitePredictor:
        """Instantiate a predictor from a pretrained checkpoint."""
        normalized_binding_type = binding_type.upper()
        if normalized_binding_type not in DEFAULT_CHECKPOINTS:
            raise ValueError(
                f"Unsupported binding_type '{binding_type}'. Expected one of: DNA, RNA."
            )

        resolved_device = _resolve_device(device)
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

        model = load_model(checkpoint_path, resolved_device)
        return cls(model=model, device=resolved_device, model_path=checkpoint_path)

    def predict_proba(
        self,
        pdb_path: str | Path,
        *,
        sequence: str | None = None,
    ) -> list[dict[str, int | str | float]]:
        """Return per-residue binding probabilities for a single PDB."""
        return run_single_inference(pdb_path, self.model, self.device, sequence=sequence)

    def predict(
        self,
        pdb_path: str | Path,
        *,
        threshold: float = 0.5,
        sequence: str | None = None,
    ) -> list[dict[str, int | str | float | bool]]:
        """Return per-residue binary predictions at a probability threshold."""
        probabilities = self.predict_proba(pdb_path, sequence=sequence)
        return [
            {
                **row,
                "is_binding": row["binding_probability"] >= threshold,
            }
            for row in probabilities
        ]
