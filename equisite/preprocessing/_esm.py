"""Internal ESM embedding utilities for inference."""

from __future__ import annotations

from typing import Any

import torch


class _ESMHolder:
    """Lazy singleton that loads ESM-2 at most once per process."""

    _model = None
    _alphabet = None
    _batch_converter = None
    _device: torch.device | None = None

    @classmethod
    def load(cls, device: torch.device) -> tuple[Any, Any, Any, torch.device]:
        """Load and cache ESM-2 on ``device``."""
        if cls._model is None:
            import esm

            print("Loading ESM-2 (esm2_t33_650M_UR50D) ...")
            cls._model, cls._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            cls._batch_converter = cls._alphabet.get_batch_converter()
            cls._model = cls._model.eval().to(device)
            cls._device = device
            print("ESM-2 ready.")

        if cls._device is None:
            raise RuntimeError("ESM holder reached an invalid state without a device.")

        return cls._model, cls._alphabet, cls._batch_converter, cls._device


def compute_esm_embeddings(sequence: str, device: torch.device) -> torch.Tensor:
    """Compute per-residue ESM-2 embeddings for ``sequence``."""
    model, _alphabet, batch_converter, esm_device = _ESMHolder.load(device)
    _, _, tokens = batch_converter([("_", sequence)])
    with torch.no_grad():
        output = model(tokens.to(esm_device), repr_layers=[33], return_contacts=False)
        representations = output["representations"][33].squeeze()
    return representations[1:-1, :].cpu()
