"""Unit tests for predict.py CLI parsing and serializers."""

from __future__ import annotations

import io
import json

import pytest


def _load_predict_helpers():
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    from equisite.model import PredictionResult
    from predict import _parse_args, _write_csv, _write_json

    return _parse_args, _write_csv, _write_json, PredictionResult


def test_parse_args_single_pdb_mode() -> None:
    """Parse single-PDB mode arguments."""
    _parse_args, _write_csv, _write_json, prediction_result = _load_predict_helpers()
    del _write_csv, _write_json, prediction_result

    args = _parse_args(["--pdb", "protein.pdb", "--type", "RNA", "--device", "cpu"])
    assert args.pdb == "protein.pdb"
    assert args.pdb_dir is None
    assert args.type == "RNA"
    assert args.device == "cpu"


def test_parse_args_batch_mode() -> None:
    """Parse batch mode arguments."""
    _parse_args, _write_csv, _write_json, prediction_result = _load_predict_helpers()
    del _write_csv, _write_json, prediction_result

    args = _parse_args(["--pdb_dir", "./pdbs", "--output", "./out", "--format", "json"])
    assert args.pdb is None
    assert args.pdb_dir == "./pdbs"
    assert args.output == "./out"
    assert args.format == "json"


def test_write_csv_output_contract() -> None:
    """Write CSV output with stable header and values."""
    _parse_args, _write_csv, _write_json, prediction_result = _load_predict_helpers()
    del _parse_args, _write_json

    results = [
        {
            "residue_index": 1,
            "chain": "A",
            "insertion_code": "",
            "residue_name": "MET",
            "binding_probability": 0.123456,
        }
    ]

    buffer = io.StringIO()
    _write_csv(prediction_result(rows=results), buffer)
    output = buffer.getvalue().strip().splitlines()

    assert output[0] == "residue_index,chain,insertion_code,residue_name,binding_probability"
    assert output[1] == "1,A,,MET,0.123456"


def test_write_json_output_contract() -> None:
    """Write JSON output with a trailing newline."""
    _parse_args, _write_csv, _write_json, prediction_result = _load_predict_helpers()
    del _parse_args, _write_csv

    results = [
        {
            "residue_index": 7,
            "chain": "B",
            "insertion_code": "A",
            "residue_name": "TYR",
            "binding_probability": 0.9,
        }
    ]

    buffer = io.StringIO()
    _write_json(prediction_result(rows=results), buffer)
    output = buffer.getvalue()

    assert output.endswith("\n")
    assert json.loads(output) == results
