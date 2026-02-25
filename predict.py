#!/usr/bin/env python3
"""
EquiSite — Predict nucleic-acid binding probabilities from PDB structures.

Usage examples
--------------
Single PDB, DNA-binding prediction:
    python predict.py --pdb protein.pdb --type DNA

Single PDB, RNA-binding, CPU-only:
    python predict.py --pdb protein.pdb --type RNA --device cpu

Batch (directory of PDBs):
    python predict.py --pdb_dir ./my_pdbs/ --type DNA --output results/

Override the protein sequence (e.g. for mutant analysis):
    python predict.py --pdb protein.pdb --type DNA --sequence "MKTLLILAS..."

Save output as JSON:
    python predict.py --pdb protein.pdb --type DNA --format json --output out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from torch_geometric.data import Data

from dataset.utils.PyPeriodicTable import PyPeriodicTable

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from dataset.utils.PyProtein import PyProtein
from model.equisite_t3_pro import EquiSite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = {
    "DNA": SCRIPT_DIR / "checkpoints" / "DNA" / "best_val.pt",
    "RNA": SCRIPT_DIR / "checkpoints" / "RNA" / "best_val.pt",
}

AA3_TO_1 = {
    "GLY": "G",
    "ALA": "A",
    "VAL": "V",
    "ILE": "I",
    "LEU": "L",
    "PHE": "F",
    "PRO": "P",
    "MET": "M",
    "TRP": "W",
    "CYS": "C",
    "SER": "S",
    "THR": "T",
    "ASN": "N",
    "GLN": "Q",
    "TYR": "Y",
    "HIS": "H",
    "ASP": "D",
    "GLU": "E",
    "LYS": "K",
    "ARG": "R",
    "UNK": "X",
}


# ======================================================================== #
#  ESM-2 singleton loader (lazy — only loaded when first needed)           #
# ======================================================================== #


class _ESMHolder:
    """Lazy singleton so ESM-2 is loaded at most once and only when needed."""

    _model = None
    _alphabet = None
    _batch_converter = None
    _device: torch.device | None = None

    @classmethod
    def load(cls, device: torch.device):
        """
        Load.

        Parameters
        ----------
        device : torch.device
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        if cls._model is None:
            import esm

            print("Loading ESM-2 (esm2_t33_650M_UR50D) …")
            cls._model, cls._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            cls._batch_converter = cls._alphabet.get_batch_converter()
            cls._model = cls._model.eval().to(device)
            cls._device = device
            print("ESM-2 ready.")
        return cls._model, cls._alphabet, cls._batch_converter, cls._device


# ======================================================================== #
#  Pipeline helpers                                                         #
# ======================================================================== #

STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
}


@dataclass(frozen=True)
class PDBResidueRecord:
    """Residue identity fields used for mapping model outputs to PDB indices."""

    chain: str
    residue_index: int
    insertion_code: str
    residue_name: str


class _ProteinOnlySelect(Select):
    """Bio.PDB selector that keeps only protein residues and canonical altlocs."""

    def accept_residue(self, residue) -> int:
        """Accept only standard amino-acid residues and UNK residues."""
        hetfield, _, _ = residue.id
        if hetfield.strip():
            return 0
        resname = residue.get_resname().strip()
        return int(is_aa(residue, standard=True) or resname == "UNK")

    def accept_atom(self, atom) -> int:
        """Keep non-disordered atoms and the primary altloc for disordered atoms."""
        altloc = atom.get_altloc().strip()
        if not atom.is_disordered():
            return 1
        return int(altloc in {"", "A", "1"})


def _load_structure(pdb_path: str | Path):
    """Parse a PDB structure using Bio.PDB."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", str(pdb_path))


def _iter_protein_residues(pdb_path: str | Path):
    """Yield protein residues from a PDB file in structural traversal order."""
    structure = _load_structure(pdb_path)
    for model in structure:
        for chain in model:
            chain_id = chain.id.strip() or "?"
            for residue in chain:
                hetfield, resseq, icode = residue.id
                if hetfield.strip():
                    continue
                resname = residue.get_resname().strip()
                if not (is_aa(residue, standard=True) or resname == "UNK"):
                    continue
                yield PDBResidueRecord(
                    chain=chain_id,
                    residue_index=int(resseq),
                    insertion_code=str(icode).strip(),
                    residue_name=resname,
                )


def _remove_hetatm(src: str | Path, dst: str | Path) -> None:
    """Write a protein-only PDB using Bio.PDB selection rules."""
    structure = _load_structure(src)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst), select=_ProteinOnlySelect())


def _extract_pdb_residue_numbers(pdb_path: str | Path) -> list[dict]:
    """Extract residue numbering fields from a protein-only PDB file."""
    return [
        {
            "chain": residue.chain,
            "resSeq": residue.residue_index,
            "iCode": residue.insertion_code,
            "resName": residue.residue_name,
        }
        for residue in _iter_protein_residues(pdb_path)
    ]


def _pdb_to_hdf5(pdb_path: str | Path, hdf5_path: str | Path) -> None:
    """Convert a cleaned PDB to internal HDF5 representation."""
    prot = PyProtein(PyPeriodicTable())
    prot.load_molecular_file(str(pdb_path))
    prot.compute_covalent_bonds()
    prot.compute_hydrogen_bonds()
    prot.save_hdf5(str(hdf5_path))


def _extract_sequence(hdf5_path: str | Path) -> tuple[str, list[str]]:
    """
    Extract the one-letter amino-acid sequence from the HDF5 file.

    Returns
    -------
    seq : str
        One-letter sequence string.
    residue_names : list[str]
        Per-residue three-letter codes (in order of appearance).
    """
    with h5py.File(str(hdf5_path), "r") as h5:
        names = [x.decode("utf-8") for x in h5["atom_residue_names"][()]]
        ids = h5["atom_amino_id"][()]

    seq: list[str] = []
    residue_names: list[str] = []
    prev = None
    for name, idx in zip(names, ids):
        if idx != prev:
            seq.append(AA3_TO_1.get(name, "X"))
            residue_names.append(name)
        prev = idx
    return "".join(seq), residue_names


def _compute_esm(seq: str, device: torch.device) -> torch.Tensor:
    """Compute ESM-2 per-residue embeddings (1280-dim) for *seq*."""
    model, _alphabet, batch_converter, esm_device = _ESMHolder.load(device)
    _, _, toks = batch_converter([("_", seq)])
    with torch.no_grad():
        res = model(toks.to(esm_device), repr_layers=[33], return_contacts=False)
        rep = res["representations"][33].squeeze()
    return rep[1:-1, :].cpu()  # strip <cls> and <eos>


# ---- DummyDB shim (borrows static methods from the dataset class) ------


class _DummyDB:
    """
    Lightweight shim that exposes the geometry helpers from
    ``dataset.DNA_Check.PBdataset.DBdataset`` without instantiating
    the full dataset machinery.
    """

    from dataset.DNA_Check.PBdataset import DBdataset as _DB

    side_chain_embs = _DB.side_chain_embs
    bb_embs = _DB.bb_embs
    get_atom_pos = _DB.get_atom_pos
    compute_dihedrals = _DB.compute_dihedrals
    _normalize = _DB._normalize


def _build_graph(
    hdf5_path: str | Path,
    seq: str,
    db: _DummyDB,
    device: torch.device,
) -> Data:
    """Build a PyG ``Data`` object from the HDF5 and sequence."""
    h5 = h5py.File(str(hdf5_path), "r")
    data = Data()

    at = h5["amino_types"][()]
    at[at == -1] = 25

    ids = h5["atom_amino_id"][()]
    names = h5["atom_names"][()]
    pos = h5["atom_pos"][()][0]

    # ESM embeddings
    data.esm_emb = _compute_esm(seq, device)

    # Geometry: atom positions per residue
    pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = db.get_atom_pos(
        at, names, ids, pos
    )

    data.side_chain_embs = db.side_chain_embs(
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h
    )
    data.side_chain_embs[torch.isnan(data.side_chain_embs)] = 0

    bb = db.bb_embs(torch.cat((pos_n.unsqueeze(1), pos_ca.unsqueeze(1), pos_c.unsqueeze(1)), 1))
    bb[torch.isnan(bb)] = 0
    data.bb_embs = bb

    data.x = torch.tensor(at).unsqueeze(1)
    data.coords_ca = pos_ca
    data.coords_n = pos_n
    data.coords_c = pos_c

    h5.close()
    return data


# ======================================================================== #
#  Model loading                                                            #
# ======================================================================== #


def _load_model(model_path: str | Path, device: torch.device) -> EquiSite:
    """Instantiate the EquiSite model and load a checkpoint."""
    ckpt = torch.load(str(model_path), map_location=device)
    model = EquiSite(
        num_blocks=4,
        hidden_channels=128,
        out_channels=1,
        cutoff=11.5,
        dropout=0.25,
        level="allatom+esm",
        args=None,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ======================================================================== #
#  Core prediction function (public API)                                    #
# ======================================================================== #


def run_single_inference(
    pdb_path: str | Path,
    model: EquiSite,
    device: torch.device,
    *,
    sequence: str | None = None,
) -> list[dict]:
    """
    Run EquiSite on a **single PDB file** and return per-residue results.

    Parameters
    ----------
    pdb_path : path-like
        Path to the input ``.pdb`` file.
    model : EquiSite
        A loaded EquiSite model (already on *device*).
    device : torch.device
        Compute device.
    sequence : str, optional
        Override the amino-acid sequence instead of extracting it from the PDB.

    Returns
    -------
    list[dict]
        Each dict has keys ``residue_index`` (PDB residue sequence number),
        ``chain``, ``residue_name``, and ``binding_probability``.
    """
    pdb_path = Path(pdb_path)
    tmp_dir = Path(pdb_path).parent / ".equisite_tmp"
    tmp_dir.mkdir(exist_ok=True)

    stem = pdb_path.stem
    clean_pdb = tmp_dir / f"{stem}_clean.pdb"
    h5_path = tmp_dir / f"{stem}.hdf5"

    try:
        # 1. Clean PDB
        _remove_hetatm(pdb_path, clean_pdb)

        # 1b. Extract PDB residue numbering from cleaned PDB
        pdb_residues = _extract_pdb_residue_numbers(clean_pdb)

        # 2. Convert to HDF5
        _pdb_to_hdf5(clean_pdb, h5_path)

        # 3. Extract / override sequence
        extracted_seq, residue_names = _extract_sequence(h5_path)
        if len(pdb_residues) != len(residue_names):
            raise ValueError(
                "Residue mapping mismatch between Bio.PDB parsing and HDF5 conversion: "
                f"{len(pdb_residues)} parsed residues vs {len(residue_names)} HDF5 residues."
            )

        seq = sequence if sequence is not None else extracted_seq
        if len(seq) != len(residue_names):
            raise ValueError(
                "Sequence length mismatch: "
                f"{len(seq)} sequence residues vs {len(residue_names)} structural residues."
            )

        # 4. Build graph
        db = _DummyDB()
        data = _build_graph(h5_path, seq, db, device).to(device)

        # 5. Predict
        with torch.no_grad():
            pred, _, _ = model(data)

        scores = pred.squeeze().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        if len(scores) != len(residue_names):
            raise ValueError(
                "Prediction length mismatch: "
                f"{len(scores)} scores vs {len(residue_names)} residue names."
            )

        # 6. Assemble results — use PDB residue numbering
        results = []
        for i, prob in enumerate(scores):
            res_name = residue_names[i]
            pdb_res = pdb_residues[i]
            res_idx = pdb_res["resSeq"]
            chain = pdb_res["chain"]
            insertion_code = pdb_res["iCode"]
            results.append(
                {
                    "residue_index": res_idx,
                    "chain": chain,
                    "insertion_code": insertion_code,
                    "residue_name": res_name,
                    "binding_probability": round(float(prob), 6),
                }
            )
        return results

    finally:
        # Cleanup temp files
        for f in (clean_pdb, h5_path):
            if f.exists():
                f.unlink()
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()


# ======================================================================== #
#  Output formatters                                                        #
# ======================================================================== #


def _write_csv(results: list[dict], dest) -> None:
    """Write results to a CSV file or stdout."""
    writer = csv.DictWriter(
        dest,
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


def _write_json(results: list[dict], dest) -> None:
    """Write results as a JSON array."""
    json.dump(results, dest, indent=2)
    dest.write("\n")


def _print_summary(results: list[dict], top_k: int, pdb_name: str) -> None:
    """Pretty-print a summary table to stderr (so stdout stays clean for piping)."""
    sorted_results = sorted(results, key=lambda r: r["binding_probability"], reverse=True)
    top = sorted_results[:top_k]

    header = f"  EquiSite — Top {min(top_k, len(results))} binding residues for {pdb_name}"
    width = max(len(header) + 4, 58)

    print("", file=sys.stderr)
    print("┌" + "─" * (width - 2) + "┐", file=sys.stderr)
    print("│" + header.ljust(width - 2) + "│", file=sys.stderr)
    print("├" + "─" * (width - 2) + "┤", file=sys.stderr)
    print(
        "│  {:<10s}  {:<10s}  {:>12s}  │".format("Index", "Residue", "Probability").ljust(width - 1)
        + "│",
        file=sys.stderr,
    )
    print("│  " + "─" * 10 + "  " + "─" * 10 + "  " + "─" * 12 + "  │", file=sys.stderr)
    for r in top:
        insertion = r.get("insertion_code", "")
        idx_label = f"{r.get('chain', '?')}:{r['residue_index']}{insertion}"
        line = "│  {:<10s}  {:<10s}  {:>12.6f}  │".format(
            idx_label, r["residue_name"], r["binding_probability"]
        )
        print(line.ljust(width - 1) + "│" if len(line) < width else line, file=sys.stderr)
    print("├" + "─" * (width - 2) + "┤", file=sys.stderr)
    total_line = f"│  Total residues: {len(results)}"
    print(total_line.ljust(width - 1) + "│", file=sys.stderr)
    print("└" + "─" * (width - 2) + "┘", file=sys.stderr)
    print("", file=sys.stderr)


# ======================================================================== #
#  CLI entry point                                                          #
# ======================================================================== #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
     parse args.

    Parameters
    ----------
    argv : list[str] | None
        Input argument.

    Returns
    -------
    argparse.Namespace
        Function output.
    """
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=textwrap.dedent("""\
            EquiSite: Predict per-residue nucleic-acid binding probabilities
            from protein PDB structures.

            Provide either --pdb (single file) or --pdb_dir (batch).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python predict.py --pdb protein.pdb --type DNA
              python predict.py --pdb protein.pdb --type RNA --device cpu
              python predict.py --pdb_dir ./pdbs/ --type DNA --output results/
              python predict.py --pdb protein.pdb --format json --output out.json
        """),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdb",
        type=str,
        metavar="FILE",
        help="Path to a single PDB file.",
    )
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
        help="Number of top-scoring residues to show in the summary (default: 20).",
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
    """
    Main.

    Parameters
    ----------
    argv : list[str] | None
        Input argument.

    Returns
    -------
    None
        Function output.
    """
    args = _parse_args(argv)

    # ---- Resolve device ------------------------------------------------
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        idx = int(args.device)
        device = torch.device(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    # ---- Resolve checkpoint -------------------------------------------
    model_path = args.model_path or str(DEFAULT_CHECKPOINTS[args.type])
    if not Path(model_path).exists():
        sys.exit(
            f"Error: checkpoint not found at {model_path}\n"
            f"Download it from the Zenodo release and place it in the expected location."
        )

    print(f"Loading model: {model_path}", file=sys.stderr)
    model = _load_model(model_path, device)

    # ---- Gather PDB files ---------------------------------------------
    if args.pdb:
        pdb_files = [Path(args.pdb)]
    else:
        pdb_dir = Path(args.pdb_dir)
        pdb_files = sorted(pdb_dir.glob("*.pdb"))
        if not pdb_files:
            sys.exit(f"Error: no .pdb files found in {pdb_dir}")
        print(f"Found {len(pdb_files)} PDB file(s).", file=sys.stderr)

    # ---- Run inference -------------------------------------------------
    write_fn = _write_csv if args.format == "csv" else _write_json
    ext = ".csv" if args.format == "csv" else ".json"

    for pdb_path in pdb_files:
        print(f"Processing: {pdb_path.name} …", file=sys.stderr)
        try:
            results = run_single_inference(
                pdb_path,
                model,
                device,
                sequence=args.sequence,
            )
        except Exception as exc:
            print(f"  ✗ Error processing {pdb_path.name}: {exc}", file=sys.stderr)
            continue

        # -- Summary to stderr --
        _print_summary(results, args.top_k, pdb_path.name)

        # -- Structured output --
        if args.output is None:
            # Single PDB → stdout
            write_fn(results, sys.stdout)
        else:
            out_path = Path(args.output)
            if len(pdb_files) > 1 or out_path.is_dir():
                # Batch mode — output directory
                out_path.mkdir(parents=True, exist_ok=True)
                dest_file = out_path / f"{pdb_path.stem}{ext}"
            else:
                # Single file
                out_path.parent.mkdir(parents=True, exist_ok=True)
                dest_file = out_path
            with open(dest_file, "w", newline="") as fh:
                write_fn(results, fh)
            print(f"  → Saved to {dest_file}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
