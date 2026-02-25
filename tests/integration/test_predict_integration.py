"""Integration test for the predict.py inference pipeline."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PDB = PROJECT_ROOT / "examples" / "3HXQ-protein.pdb"
DNA_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "DNA" / "best_val.pt"
REFERENCE_CSV = PROJECT_ROOT / "examples" / "result.csv"


def _read_reference_row_count(csv_path: Path) -> int:
    """Return the number of prediction rows in a reference CSV file."""
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


@pytest.fixture(scope="session")
def loaded_model() -> tuple[Any, Any]:
    """Load the DNA model once for the integration test session."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("esm")
    if not EXAMPLE_PDB.exists():
        pytest.skip(f"Example PDB not found: {EXAMPLE_PDB}")
    if not DNA_CHECKPOINT.exists():
        pytest.skip(f"DNA checkpoint not found: {DNA_CHECKPOINT}")

    from predict import _load_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(DNA_CHECKPOINT, device)
    return model, device


def test_predict_pipeline_contract(loaded_model: tuple[Any, Any]) -> None:
    """Run end-to-end inference and validate output mapping and probability contract."""
    from predict import run_single_inference

    model, device = loaded_model
    results = run_single_inference(EXAMPLE_PDB, model, device)

    assert results, "Predict pipeline returned no residue-level predictions."
    assert len(results) == _read_reference_row_count(REFERENCE_CSV)

    expected_fields = {
        "residue_index",
        "chain",
        "insertion_code",
        "residue_name",
        "binding_probability",
    }
    for row in results:
        assert set(row) == expected_fields
        assert isinstance(row["residue_index"], int)
        assert isinstance(row["chain"], str) and row["chain"]
        assert isinstance(row["insertion_code"], str)
        assert isinstance(row["residue_name"], str) and len(row["residue_name"]) == 3
        probability = row["binding_probability"]
        assert isinstance(probability, float)
        assert math.isfinite(probability)
        assert 0.0 <= probability <= 1.0
