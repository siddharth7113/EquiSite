# EquiSite

**Multi-Scale Equivariant Graph Learning for Robust Nucleic Acid Binding Site Prediction**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18617058.svg)](https://doi.org/10.5281/zenodo.18617058)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EquiSite is an SE(3)-equivariant geometric graph neural network that predicts
**protein–DNA** and **protein–RNA** binding sites at per-residue resolution.
It explicitly models multi-scale protein geometry (backbone *and* side-chain
orientations) and **eliminates the need for evolutionary profiles (MSAs)**,
making it fast enough for high-throughput use on AlphaFold2 models.

<!-- ![Graphical Abstract](graphical_abstract.png) -->
<img src="graphical_abstract.png" alt="Graphical Abstract" width="40%">

---

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage — `predict.py`](#detailed-usage--predictpy)
- [Training](#training)
- [Checkpoints](#checkpoints)
- [Repository Structure](#repository-structure)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## How It Works

```
PDB file ──► Clean HETATM ──► HDF5 ──► ESM-2 embeddings ──► EquiSite GNN ──► Per-residue
             (strip ligands)   (atom    + Backbone/side-chain    (SE(3)-        binding
                                coords)  geometry features       equivariant)    probabilities
```

1. **PDB preprocessing** — HETATM records (ligands, waters, etc.) are stripped;
   the clean structure is converted to an internal HDF5 representation containing
   atom positions, residue types, and covalent/hydrogen bond information.
2. **Sequence embeddings** — The protein sequence is passed through
   [ESM-2 (650M)](https://github.com/facebookresearch/esm) to produce 1280-dim
   per-residue embeddings.
3. **Geometric features** — Backbone dihedral angles (φ, ψ, ω) and side-chain
   torsion angles (χ₁–χ₄) are computed from atom coordinates.
4. **Graph construction** — A radius graph (default cutoff 11.5 Å) over Cα atoms
   is built, with spherical harmonic edge features.
5. **EquiSite model** — A hybrid architecture combining an Equiformer backbone
   (SE(3)-equivariant attention) with geometric message passing. Outputs a
   per-residue softmax probability of being a nucleic-acid binding site.

---

## Installation

### Prerequisites

- **Python ≥ 3.9, < 3.12**
- **CUDA-capable GPU** (recommended; CPU works but is slow for ESM-2)
- **[PyTorch](https://pytorch.org/get-started/locally/)** with CUDA support

### Option A — uv (recommended)

```bash
# 1. Create and activate a virtual environment
uv venv --python 3.11
source .venv/bin/activate

# 2. Install PyTorch with CUDA (example for CUDA 12.4)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install PyTorch Geometric extensions
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 4. Install EquiSite and remaining dependencies
uv pip install -e ".[dev]"
```

> **Note:** Adjust the CUDA version in steps 2–3 to match your driver.
> Run `nvidia-smi` to check your CUDA version. Your driver's CUDA version is
> backward-compatible, so e.g. a CUDA 13.x driver can use `cu124` wheels.

---

## Quick Start

### Predict DNA-binding residues for a single protein

```bash
python predict.py --pdb my_protein.pdb --type DNA
```

This will:
1. Load the pretrained DNA checkpoint
2. Process the PDB through the full pipeline
3. Print a **summary table** of top binding residues to the terminal
4. Write a CSV with *all* per-residue probabilities to stdout

### Save results to a file

```bash
python predict.py --pdb my_protein.pdb --type DNA --output results.csv
```

### Predict RNA-binding residues

```bash
python predict.py --pdb my_protein.pdb --type RNA --output rna_results.csv
```

### Run on CPU (no GPU)

```bash
python predict.py --pdb my_protein.pdb --type DNA --device cpu
```

### Batch mode (directory of PDBs)

```bash
python predict.py --pdb_dir ./my_pdbs/ --type DNA --output ./results/
```

Each PDB gets its own output file in `./results/`.

---

## Detailed Usage — `predict.py`

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--pdb` | `FILE` | — | Path to a single `.pdb` file *(mutually exclusive with `--pdb_dir`)* |
| `--pdb_dir` | `DIR` | — | Directory containing `.pdb` files (batch mode) |
| `--type` | `DNA\|RNA` | `DNA` | Binding type to predict |
| `--model_path` | `FILE` | auto | Override the default checkpoint |
| `--output` / `-o` | `PATH` | stdout | Output file (single) or directory (batch) |
| `--format` | `csv\|json` | `csv` | Output format |
| `--top_k` | `int` | `20` | Number of top residues shown in summary |
| `--device` | `str` | `0` | `"cpu"` or CUDA device index |
| `--sequence` | `str` | — | Override protein sequence (e.g. for mutant studies) |

### Output Format

**CSV** (default):

```csv
residue_index,residue_name,binding_probability
1,MET,0.012345
2,LYS,0.234567
3,THR,0.891234
...
```

**JSON** (`--format json`):

```json
[
  {"residue_index": 1, "residue_name": "MET", "binding_probability": 0.012345},
  {"residue_index": 2, "residue_name": "LYS", "binding_probability": 0.234567},
  ...
]
```

### Programmatic Usage

You can also use the public Python inference API:

```python
from equisite.model import EquiSitePipeline

pipeline = EquiSitePipeline.from_pretrained(binding_type="DNA", device="0")
results = pipeline.predict_proba("my_protein.pdb")
for r in results:
    if r["binding_probability"] > 0.5:
        print(f"Residue {r['residue_index']} ({r['residue_name']}): "
              f"{r['binding_probability']:.4f}")

results.to_csv("predictions.csv")
```

---

## Training

Train EquiSite from scratch on your own data:

```bash
python train.py \
    --dataset DNA_Check \
    --dataset_path dataset/ \
    --epochs 200 \
    --batch_size 4 \
    --eval_batch_size 2 \
    --hidden_channels 128 \
    --num_blocks 4 \
    --cutoff 11.5 \
    --level allatom+esm
```

### Key Arguments

| Argument | Description |
|---|---|
| `--dataset` | Dataset name: `DNA_Check`, `RNA_Check`, etc. |
| `--dataset_path` | Root path containing dataset directories |
| `--epochs` | Number of training epochs |
| `--batch_size` | Training batch size |
| `--level` | Model level: `aminoacid`, `backbone`, `allatom`, `backbone+esm`, or `allatom+esm` |
| `--cutoff` | Radius-graph cutoff distance (Å) |

Checkpoints and TensorBoard logs are saved to `./saves/`. The best model
(by validation ROC-AUC) is saved as `best.pt`.

---

## Checkpoints

Pretrained weights are not included in the repository:

| Checkpoint | Task | Path |
|---|---|---|
| DNA (573 train) | DNA-binding | `checkpoints/DNA/best_val.pt` |
| DNA (181 test) | DNA-binding | `checkpoints/DNA_181/best_val.pt` |
| RNA | RNA-binding | `checkpoints/RNA/best_val.pt` |

Checkpoints and full evaluation outputs are available on
[Zenodo (DOI: 10.5281/zenodo.18617058)](https://doi.org/10.5281/zenodo.18617058).

---

## Repository Structure

```
EquiSite/
├── predict.py              # ← User-friendly inference CLI (start here!)
├── train.py                # Training script
├── notebooks/              # Example Jupyter workflows
│   └── inference.ipynb
├── pyproject.toml          # pip-installable package metadata
├── checkpoints/            # Pretrained weights (DNA / RNA)
├── examples/               # Sample PDBs and output CSVs
│
├── equisite/               # Public Python inference API
│   ├── __init__.py
│   ├── model/              # Public model + pipeline API
│   │   ├── _model.py
│   │   ├── _pipeline.py
│   │   ├── _result.py
│   │   └── layers/         # Decomposed model building blocks
│   ├── utils/              # Shared loss/padding/metrics helpers
│   ├── data/
│   └── datasets/
│   └── _*.py               # Other private inference internals
│
├── dataset/
│   ├── DNA_Check/          # DNA dataset + data loader (PBdataset.py)
│   ├── RNA_Check/          # RNA dataset
│   ├── PATP/, PCA/, …      # Additional benchmark datasets
│   └── utils/              # PDB/HDF5 parsing (PyProtein, PyPeriodicTable)
```

---

## FAQ & Troubleshooting

### "CUDA out of memory" when running `predict.py`

ESM-2 (650M parameters) requires ~2–3 GB of GPU memory. The full pipeline
(ESM-2 + EquiSite) typically needs **≥ 6 GB** for a single protein.

- Use `--device cpu` if your GPU is too small (slower but always works).
- For very large proteins (> 1000 residues), CPU mode may be necessary.

### PyTorch Geometric installation errors

PyG extensions (`torch-scatter`, `torch-sparse`, etc.) must match your exact
PyTorch and CUDA versions. Follow the [official PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html):

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

Replace `torch-2.6.0+cu124` with your actual PyTorch version and CUDA tag.

### `ModuleNotFoundError: No module named 'equisite'` or `'dataset'`

Make sure you installed the package in development mode:

```bash
uv pip install -e .
```

This registers `equisite/` and `dataset/` as importable packages.

### How do I use this for docking (e.g. HADDOCK)?

Use `predict.py` to identify high-probability binding residues, then supply
those residue indices as "active residues" in your docking setup. A reasonable
threshold is probability > 0.5, but you may adjust based on your use case.

---

## Citation

If you use EquiSite in your research, please cite:

```bibtex
@article{equisite2024,
  title   = {EquiSite: Multi-Scale Equivariant Graph Learning for Robust
             Nucleic Acid Binding Site Prediction},
  year    = {2024},
  doi     = {10.5281/zenodo.18617058}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
