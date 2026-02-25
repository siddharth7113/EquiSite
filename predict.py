#!/usr/bin/env python3
"""EquiSite CLI for nucleic-acid binding prediction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from typing import TextIO

from equisite import EquiSitePredictor


def _write_csv(results: list[dict[str, int | str | float]], destination: TextIO) -> None:
    """Write predictions to CSV."""
    writer = csv.DictWriter(
        destination,
        fieldnames=[
            "residue_index",
            "chain",
            "insertion_code",
            "residue_name",
            "binding_probability",
        ],
    )
    writer.writeheader()
    writer.writerows(results)


def _write_json(results: list[dict[str, int | str | float]], destination: TextIO) -> None:
    """Write predictions to JSON."""
    json.dump(results, destination, indent=2)
    destination.write("\n")


def _print_summary(
    results: list[dict[str, int | str | float]],
    top_k: int,
    pdb_name: str,
) -> None:
    """Print top-ranked residues to stderr."""
    sorted_results = sorted(
        results, key=lambda row: float(row["binding_probability"]), reverse=True
    )
    top_results = sorted_results[:top_k]

    print("", file=sys.stderr)
    print(
        f"EquiSite - Top {min(top_k, len(results))} binding residues for {pdb_name}",
        file=sys.stderr,
    )
    print("Index       Residue        Probability", file=sys.stderr)
    print("--------------------------------------", file=sys.stderr)

    for row in top_results:
        insertion_code = str(row.get("insertion_code", ""))
        index_label = f"{row.get('chain', '?')}:{row['residue_index']}{insertion_code}"
        print(
            f"{index_label:<10}  {row['residue_name']:<10}  {float(row['binding_probability']):>11.6f}",
            file=sys.stderr,
        )

    print("--------------------------------------", file=sys.stderr)
    print(f"Total residues: {len(results)}", file=sys.stderr)
    print("", file=sys.stderr)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=textwrap.dedent(
            """\
            EquiSite: Predict per-residue nucleic-acid binding probabilities
            from protein PDB structures.

            Provide either --pdb (single file) or --pdb_dir (batch).
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python predict.py --pdb protein.pdb --type DNA
              python predict.py --pdb protein.pdb --type RNA --device cpu
              python predict.py --pdb_dir ./pdbs/ --type DNA --output results/
              python predict.py --pdb protein.pdb --format json --output out.json
            """
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdb", type=str, metavar="FILE", help="Path to a single PDB file.")
    input_group.add_argument(
        "--pdb_dir",
        type=str,
        metavar="DIR",
        help="Path to a directory of PDB files (batch mode).",
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["DNA", "RNA"],
        default="DNA",
        help="Binding type to predict (default: DNA).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        metavar="FILE",
        help="Override the default checkpoint path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output path. For single-PDB: a file path (default: stdout). "
            "For batch: a directory (created if needed)."
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top-scoring residues shown in summary (default: 20).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='Device: "cpu" or CUDA device index (default: "0").',
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Override the protein sequence instead of extracting from PDB.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = _parse_args(argv)
    try:
        predictor = EquiSitePredictor.from_pretrained(
            binding_type=args.type,
            model_path=args.model_path,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")

    print(f"Using device: {predictor.device}", file=sys.stderr)
    if predictor.model_path is not None:
        print(f"Loading model: {predictor.model_path}", file=sys.stderr)

    if args.pdb:
        pdb_files = [Path(args.pdb)]
    else:
        pdb_dir = Path(args.pdb_dir)
        pdb_files = sorted(pdb_dir.glob("*.pdb"))
        if not pdb_files:
            sys.exit(f"Error: no .pdb files found in {pdb_dir}")
        print(f"Found {len(pdb_files)} PDB file(s).", file=sys.stderr)

    write_fn = _write_csv if args.format == "csv" else _write_json
    extension = ".csv" if args.format == "csv" else ".json"

    for pdb_path in pdb_files:
        print(f"Processing: {pdb_path.name} ...", file=sys.stderr)
        try:
            results = predictor.predict_proba(pdb_path, sequence=args.sequence)
        except Exception as exc:  # pragma: no cover - this is CLI error handling.
            print(f"  x Error processing {pdb_path.name}: {exc}", file=sys.stderr)
            continue

        _print_summary(results, args.top_k, pdb_path.name)

        if args.output is None:
            write_fn(results, sys.stdout)
            continue

        output_path = Path(args.output)
        if len(pdb_files) > 1 or output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            destination_file = output_path / f"{pdb_path.stem}{extension}"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            destination_file = output_path

        with open(destination_file, "w", newline="") as handle:
            write_fn(results, handle)
        print(f"  -> Saved to {destination_file}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
