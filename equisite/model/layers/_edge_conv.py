"""Graph convolution primitives for edge-aware EquiSite message passing."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul

from ._linear import Linear


class EdgeGraphConv(MessagePassing):
    """Edge-weighted graph convolution with Hadamard message scaling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolution layers and parameters."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        size: Any = None,
    ) -> torch.Tensor:
        """Run the message passing step."""
        pair = (x, x)
        out = self.propagate(edge_index, x=pair, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(pair[1])

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """Compute edge-weighted messages."""
        return edge_weight * x_j

    def message_and_aggregate(
        self, adj_t: Any, x: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse sparse message and aggregation for prebuilt adjacency tensors."""
        return matmul(adj_t, x[0], reduce=self.aggr)
