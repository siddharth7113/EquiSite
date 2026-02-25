"""Unit tests for the public EquiSitePredictor API."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_from_pretrained_raises_on_missing_checkpoint(tmp_path: Path) -> None:
    """Fail fast when a checkpoint path does not exist."""
    pytest.importorskip("torch")
    from equisite.predictor import EquiSitePredictor

    missing_checkpoint = tmp_path / "does_not_exist.pt"
    try:
        EquiSitePredictor.from_pretrained(model_path=missing_checkpoint)
    except FileNotFoundError as error:
        assert str(missing_checkpoint) in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected FileNotFoundError for missing checkpoint path")


def test_predict_adds_binary_binding_flag() -> None:
    """Convert probability rows to thresholded binary predictions."""
    torch = pytest.importorskip("torch")
    from equisite.predictor import EquiSitePredictor

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    predictor = EquiSitePredictor(model=_DummyModel(), device=torch.device("cpu"))

    def _fake_predict_proba(*args, **kwargs):
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

    predictor.predict_proba = _fake_predict_proba  # type: ignore[method-assign]
    predictions = predictor.predict("protein.pdb", threshold=0.5)

    assert predictions[0]["is_binding"] is False
    assert predictions[1]["is_binding"] is True
