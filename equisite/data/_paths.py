"""Data path dataclasses for programmatic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Canonical dataset path layout rooted at ``root``."""

    root: Path

    @property
    def dataset_root(self) -> Path:
        """Return the dataset root directory."""
        return self.root / "dataset"

    @property
    def checkpoint_root(self) -> Path:
        """Return the checkpoint root directory."""
        return self.root / "checkpoints"
