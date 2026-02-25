"""Dataset configuration dataclasses for EquiSite workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset descriptor used by training and evaluation pipelines."""

    name: str
    root: Path
    split: str

    @property
    def split_file(self) -> Path:
        """Return expected split filename under ``root``."""
        return self.root / f"{self.split}.txt"
