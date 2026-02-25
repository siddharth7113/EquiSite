"""Unit tests for the public equisite.model API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _build_dummy_model(torch):
    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    return _DummyModel()


def test_pipeline_predict_proba_returns_result_object(monkeypatch) -> None:
    """Return PredictionResult from the public pipeline API."""
    torch = pytest.importorskip("torch")
    from equisite.model import EquiSitePipeline

    dummy_model = _build_dummy_model(torch)
    pipeline = EquiSitePipeline(model=dummy_model, device="cpu")

    def _fake_run_single_inference(*args, **kwargs):
        del args, kwargs
        return [
            {
                "residue_index": 1,
                "chain": "A",
                "insertion_code": "",
                "residue_name": "MET",
                "binding_probability": 0.42,
            }
        ]

    monkeypatch.setattr("equisite.model._pipeline.run_single_inference", _fake_run_single_inference)
    result = pipeline.predict_proba("protein.pdb")

    assert len(result) == 1
    assert result[0]["binding_probability"] == 0.42


def test_pipeline_predict_adds_binary_flag(monkeypatch) -> None:
    """Return BinaryPredictionResult with thresholded labels."""
    torch = pytest.importorskip("torch")
    from equisite.model import EquiSitePipeline

    dummy_model = _build_dummy_model(torch)
    pipeline = EquiSitePipeline(model=dummy_model, device="cpu")

    def _fake_run_single_inference(*args, **kwargs):
        del args, kwargs
        return [
            {
                "residue_index": 1,
                "chain": "A",
                "insertion_code": "",
                "residue_name": "MET",
                "binding_probability": 0.2,
            },
            {
                "residue_index": 2,
                "chain": "A",
                "insertion_code": "",
                "residue_name": "THR",
                "binding_probability": 0.8,
            },
        ]

    monkeypatch.setattr("equisite.model._pipeline.run_single_inference", _fake_run_single_inference)
    result = pipeline.predict("protein.pdb", threshold=0.5)

    assert result[0]["is_binding"] is False
    assert result[1]["is_binding"] is True


def test_prediction_result_writes_csv_and_json(tmp_path: Path) -> None:
    """Persist prediction results via helper export methods."""
    from equisite.model import PredictionResult

    result = PredictionResult(
        rows=[
            {
                "residue_index": 1,
                "chain": "A",
                "insertion_code": "",
                "residue_name": "MET",
                "binding_probability": 0.1,
            }
        ]
    )

    csv_path = tmp_path / "result.csv"
    json_path = tmp_path / "result.json"

    result.to_csv(csv_path)
    result.to_json(json_path)

    csv_text = csv_path.read_text().strip().splitlines()
    assert csv_text[0] == "residue_index,chain,insertion_code,residue_name,binding_probability"
    assert csv_text[1] == "1,A,,MET,0.1"

    loaded_json = json.loads(json_path.read_text())
    assert loaded_json == result.rows
