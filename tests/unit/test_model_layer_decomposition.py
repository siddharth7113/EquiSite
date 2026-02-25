"""Unit tests for decomposed model layer modules."""

from __future__ import annotations

import pytest


def test_legacy_model_uses_decomposed_linear_layers() -> None:
    """Expose decomposed layer classes through the legacy model module."""
    pytest.importorskip("torch")

    from equisite.model.layers import EdgeGraphConv, InteractionBlock, Linear, TwoLinear, swish
    from model.equisite_t3_pro import (
        EdgeGraphConv as LegacyEdgeGraphConv,
        InteractionBlock as LegacyInteractionBlock,
        Linear as LegacyLinear,
        TwoLinear as LegacyTwoLinear,
        swish as legacy_swish,
    )

    assert LegacyLinear is Linear
    assert LegacyTwoLinear is TwoLinear
    assert LegacyEdgeGraphConv is EdgeGraphConv
    assert LegacyInteractionBlock is InteractionBlock
    assert legacy_swish is swish


def test_two_linear_forward_shape() -> None:
    """Run a lightweight shape contract check for extracted layer blocks."""
    torch = pytest.importorskip("torch")
    from equisite.model.layers import TwoLinear

    layer = TwoLinear(in_channels=6, middle_channels=4, out_channels=3, bias=True, act=True)
    output = layer(torch.randn(2, 6))
    assert output.shape == (2, 3)
