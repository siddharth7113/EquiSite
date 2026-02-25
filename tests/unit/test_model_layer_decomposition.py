"""Unit tests for decomposed model layer modules."""

from __future__ import annotations

import pytest


def test_core_model_uses_decomposed_linear_layers() -> None:
    """Expose decomposed layer classes through the core model module."""
    pytest.importorskip("torch")

    from equisite.model.equisite_t3_pro import (
        EdgeGraphConv as CoreEdgeGraphConv,
        InteractionBlock as CoreInteractionBlock,
        Linear as CoreLinear,
        TwoLinear as CoreTwoLinear,
        swish as core_swish,
    )
    from equisite.model.layers import EdgeGraphConv, InteractionBlock, Linear, TwoLinear, swish

    assert CoreLinear is Linear
    assert CoreTwoLinear is TwoLinear
    assert CoreEdgeGraphConv is EdgeGraphConv
    assert CoreInteractionBlock is InteractionBlock
    assert core_swish is swish


def test_two_linear_forward_shape() -> None:
    """Run a lightweight shape contract check for extracted layer blocks."""
    torch = pytest.importorskip("torch")
    from equisite.model.layers import TwoLinear

    layer = TwoLinear(in_channels=6, middle_channels=4, out_channels=3, bias=True, act=True)
    output = layer(torch.randn(2, 6))
    assert output.shape == (2, 3)
