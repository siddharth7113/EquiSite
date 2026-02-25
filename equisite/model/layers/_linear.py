"""Linear layer primitives for EquiSite models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import inits

from ._activations import swish


class Linear(torch.nn.Module):
    """Linear module with configurable weight initialization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: str = "glorot",
    ) -> None:
        """Initialize the linear layer."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer weights and optional bias."""
        if self.weight_initializer == "glorot":
            inits.glorot(self.weight)
        elif self.weight_initializer == "zeros":
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation."""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """Two linear layers with optional swish activations in between."""

    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        bias: bool = False,
        act: bool = False,
    ) -> None:
        """Initialize the two-layer perceptron block."""
        super().__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self) -> None:
        """Reset internal linear layer parameters."""
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through both linear layers."""
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x
