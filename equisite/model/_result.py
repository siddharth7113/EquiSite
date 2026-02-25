"""Prediction result containers for the public model API."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .._types import BinaryPredictionRow, PredictionRow


@dataclass(frozen=True)
class PredictionResult(Sequence[PredictionRow]):
    """Container for per-residue probability predictions."""

    rows: list[PredictionRow]

    def __getitem__(self, index: int) -> PredictionRow:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[PredictionRow]:
        return iter(self.rows)

    def to_csv(self, path: str | Path) -> None:
        """Write prediction rows to a CSV file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "residue_index",
                    "chain",
                    "insertion_code",
                    "residue_name",
                    "binding_probability",
                ],
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write prediction rows to a JSON file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as handle:
            json.dump(self.rows, handle, indent=indent)
            handle.write("\n")

    def to_list(self) -> list[PredictionRow]:
        """Return a copy of prediction rows."""
        return list(self.rows)


@dataclass(frozen=True)
class BinaryPredictionResult(Sequence[BinaryPredictionRow]):
    """Container for thresholded binary predictions."""

    rows: list[BinaryPredictionRow]

    def __getitem__(self, index: int) -> BinaryPredictionRow:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[BinaryPredictionRow]:
        return iter(self.rows)

    def to_list(self) -> list[BinaryPredictionRow]:
        """Return a copy of binary prediction rows."""
        return list(self.rows)
