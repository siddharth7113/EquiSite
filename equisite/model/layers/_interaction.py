"""Interaction block used by EquiSite neighborhood processing."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ._activations import swish
from ._edge_conv import EdgeGraphConv
from ._linear import Linear, TwoLinear


class InteractionBlock(torch.nn.Module):
    """Multi-branch interaction block combining geometric edge features."""

    def __init__(
        self,
        hidden_channels: int,
        output_channels: int,
        num_radial: int,
        num_spherical: int,
        num_layers: int,
        mid_emb: int,
        act: Callable[[torch.Tensor], torch.Tensor] = swish,
        num_pos_emb: int = 16,
        dropout: float = 0,
        level: str = "allatom",
    ) -> None:
        """Initialize the interaction block modules."""
        super().__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)

        self.lin_feature0 = TwoLinear(num_radial * num_spherical**2, mid_emb, hidden_channels)
        if level == "aminoacid":
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif (
            level == "backbone"
            or level == "allatom"
            or level == "backbone+esm"
            or level == "allatom+esm"
        ):
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(3 * hidden_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters in all submodules."""
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        feature0: torch.Tensor,
        feature1: torch.Tensor,
        pos_emb: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the interaction block and return updated node embeddings."""
        del batch

        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))

        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)
        h0 = self.lin0(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        h = torch.cat((h0, h1, h2), 1)
        for lin in self.lins_cat:
            h = self.act(lin(h))

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(lin(h))
        h = self.final(h)
        return h
