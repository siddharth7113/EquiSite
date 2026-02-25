"""Unit tests for internal device resolution helpers."""

from __future__ import annotations

import pytest


def test_resolve_device_cpu_string() -> None:
    """Resolve explicit CPU input."""
    pytest.importorskip("torch")
    from equisite._device import resolve_device

    device = resolve_device("cpu")
    assert str(device) == "cpu"


def test_resolve_device_none_returns_torch_device() -> None:
    """Resolve default device when input is None."""
    pytest.importorskip("torch")
    from equisite._device import resolve_device

    device = resolve_device(None)
    assert device.type in {"cpu", "cuda"}
