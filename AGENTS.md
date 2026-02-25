# AGENTS.md — EquiSite

## Purpose

This file is for coding agents working in this repository. Follow these
instructions before making code changes.

## Project Snapshot

EquiSite predicts protein nucleic-acid binding residues using:
- PyTorch + PyTorch Geometric
- ESM-2 residue embeddings
- SE(3)-equivariant graph attention (Equiformer-style backbone)

## Current Repository Structure

```
EquiSite/
├── predict.py                 # Main inference CLI
├── train.py                   # Training entry point
├── notebooks/                 # Example notebook workflows
│   └── inference.ipynb
├── pyproject.toml             # Packaging + black/ruff configuration
├── checkpoints/               # Pretrained model weights
│   ├── DNA/best_val.pt
│   ├── DNA_181/best_val.pt
│   └── RNA/best_val.pt
├── examples/                  # Sample PDBs and output CSV files
├── equisite/                  # Public Python bindings + private inference modules
│   ├── __init__.py
│   ├── model/
│   │   ├── _model.py
│   │   ├── _pipeline.py
│   │   ├── _result.py
│   │   ├── equisite_t3_pro.py
│   │   ├── features_equi_t3_pro.py
│   │   ├── layers/
│   │   └── nets/
│   ├── utils/
│   ├── data/
│   ├── datasets/
│   └── _*.py
├── dataset/
│   ├── __init__.py
│   ├── DNA_Check/, RNA_Check/, PATP/, PCA/, PHEM/, PMG/, PMN/
│   │   ├── __init__.py
│   │   ├── PBdataset.py
│   │   └── *_Train.txt / *_Test.txt
│   └── utils/
│       ├── __init__.py
│       └── PyProtein.py, PyMolecule.py, PyMolIO.py, PyPeriodicTable.py
```

## Environment Management (uv only)

Conda is intentionally removed from this repository workflow.

```bash
# 1) Create venv
uv venv --python 3.11
source .venv/bin/activate

# 2) Install PyTorch CUDA wheels (example: cu124)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3) Install PyG extension wheels matching torch/cuda
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv   -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# 4) Install project + dev tools
uv pip install -e ".[dev]"
```

## Run Commands

```bash
# Inference (single protein)
python predict.py --pdb examples/3HXQ-protein.pdb --type DNA

# Inference (batch)
python predict.py --pdb_dir ./my_pdbs --type RNA --output ./results

# Training
python train.py --dataset DNA_Check --dataset_path dataset/ --epochs 200
```

## Lint and Formatting Rules

Linting and formatting are configured in `pyproject.toml`.

```bash
black .
ruff check .
ruff check . --fix
```

Configured defaults:
- Black line length: `100`
- Ruff line length: `100`
- Ruff target version: `py39`
- Ruff select: `E`, `F`, `W`, `I`, `UP`, `B`
- Ruff ignore: `E203`, `E501`, `E741`, `B005`, `B006`, `B007`, `B023`, `B904`, `UP031`

## Testing Guidance

There is currently no formal test suite in this repository.

If you add tests:
- Place them under `tests/`
- Use `pytest`
- Single test command:
  - `pytest tests/test_foo.py::test_bar -v`

## Code Style Expectations

### Docstrings

Use numpydoc style for public classes, functions, and methods.

Principles:
- Be concise and precise.
- Explain behavior and contracts, not implementation trivia.
- Include `Parameters` and `Returns` where applicable.
- Use `Raises` only when exceptions are part of API behavior.

### Imports

- Order imports: stdlib -> third-party -> local.
- Use explicit imports (no `from x import *`).
- Use relative imports for same-package modules.

### Naming

- Classes: `PascalCase`
- Functions/methods/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: leading underscore (`_helper_name`)

### Typing

- Add type hints to new or heavily edited code.
- Use modern unions (`A | B`) where supported.

### Error Handling

- Catch specific exception types when possible.
- Avoid broad `except Exception` unless re-raising with context.
- Use deterministic cleanup for temp files/resources.

### Comments

- Keep comments in English.
- Only keep comments that clarify non-obvious logic.
- Prefer docstrings for API-level behavior.

## ML-Specific Conventions

- Keep tensor shape assumptions explicit in docstrings/comments when relevant.
- Avoid changing checkpoint compatibility unless requested.
- Do not hardcode CUDA-only logic in new code paths.
- Avoid expensive module-level side effects when adding new components.

## Pull Request / Commit Expectations

When making substantial edits:
- Separate structure, docs, and style changes into different commits when possible.
- Use descriptive commit messages that explain intent.
- Include touched areas and rationale in commit body for easier review.

Suggested commit style:
- `chore: ...` for repository and structure changes
- `docs: ...` for docstrings and documentation
- `style: ...` for formatting/lint-only updates
