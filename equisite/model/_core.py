"""
This is an implementation of EquiSite model

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding
from torch_geometric.nn import radius_graph

from ._features import d_angle_emb, d_theta_phi_emb
from .layers import EdgeGraphConv, InteractionBlock, Linear, TwoLinear, swish

__all__ = [
    "swish",
    "Linear",
    "TwoLinear",
    "EdgeGraphConv",
    "InteractionBlock",
    "EquiSite",
    "batchgraph2batch",
]

num_aa_type = 26
num_side_chain_embs = 8
num_bb_embs = 6
num_esm_embs = 1280


class EquiSite(nn.Module):
    """
    Args:

    """

    def __init__(
        self,
        *,
        args: Any = None,
        level: str = "allatom+esm",
        num_blocks: int = 4,
        hidden_channels: int = 128,
        out_channels: int = 1,
        mid_emb: int = 64,
        num_radial: int = 6,
        num_spherical: int = 3,
        cutoff: float = 11.5,
        max_num_neighbors: int = 32,
        int_emb_layers: int = 3,
        out_layers: int = 2,
        num_pos_emb: int = 16,
        dropout: float = 0.25,
        data_augment_eachlayer: bool = False,
        euler_noise: bool = False,
    ) -> None:
        """
        Initialize EquiSite.

        Parameters
        ----------
        args : Any
            Input argument.
        level : Any
            Input argument.
        num_blocks : Any
            Input argument.
        hidden_channels : Any
            Input argument.
        out_channels : Any
            Input argument.
        mid_emb : Any
            Input argument.
        num_radial : Any
            Input argument.
        num_spherical : Any
            Input argument.
        cutoff : Any
            Input argument.
        max_num_neighbors : Any
            Input argument.
        int_emb_layers : Any
            Input argument.
        out_layers : Any
            Input argument.
        num_pos_emb : Any
            Input argument.
        dropout : Any
            Input argument.
        data_augment_eachlayer : Any
            Input argument.
        euler_noise : Any
            Input argument.

        """
        del num_blocks, int_emb_layers

        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.act = swish

        self.feature0 = d_theta_phi_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )
        self.feature1 = d_angle_emb(
            num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff
        )

        if level == "aminoacid":
            self.embedding = Embedding(num_aa_type, hidden_channels)
        elif level == "backbone":
            self.embedding = torch.nn.Linear(num_aa_type + num_bb_embs, hidden_channels)
        elif level == "allatom":
            self.embedding = torch.nn.Linear(
                num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels
            )
        elif level == "backbone+esm":
            self.embedding = torch.nn.Linear(
                num_aa_type + num_bb_embs + num_esm_embs, hidden_channels
            )
        elif level == "allatom+esm":
            self.embedding = torch.nn.Linear(
                num_aa_type + num_bb_embs + num_side_chain_embs, hidden_channels // 2
            )
            self.embedding_esm = torch.nn.Linear(num_esm_embs, hidden_channels // 2)
        else:
            print("No supported model!")

        if "equiformer":
            from .nets.graph_attention_transformer_t3_pro import GraphAttentionTransformer

            self.model_E = GraphAttentionTransformer(
                irreps_in=None,
                irreps_node_embedding="128x0e+64x1e+32x2e",
                num_layers=6,
                irreps_node_attr="1x0e",
                irreps_sh="1x0e+1x1e+1x2e",
                max_radius=self.cutoff,
                number_of_basis=32,
                fc_neurons=[64, 64],
                irreps_feature="512x0e",
                irreps_head="32x0e+16x1e+8x2e",
                num_heads=4,
                irreps_pre_attn=None,
                rescale_degree=False,
                nonlinear_message=True,
                irreps_mlp_mid="384x0e+192x1e+96x2e",
                norm_layer="layer",
                alpha_drop=0.2,
                proj_drop=0.0,
                out_drop=0.0,
                drop_path_rate=0.0,
            )

        self.lins_out = torch.nn.ModuleList()
        self.lins_node_out = torch.nn.ModuleList()
        for _ in range(out_layers - 1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
            self.lins_node_out.append(Linear(hidden_channels, 32))
        self.lins_out.append(Linear(hidden_channels, 32))
        self.lin_out = Linear(32, out_channels)
        self.lin_node_out = Linear(32, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        self.args = args

    def reset_parameters(self) -> None:
        """
        Reset parameters.

        Returns
        -------
        Any
            Function output.
        """
        self.embedding.reset_parameters()
        # for interaction in self.interaction_blocks:
        #     interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index: torch.Tensor, num_pos_emb: int = 16) -> torch.Tensor:
        # From https://github.com/jingraham/neurips19-graph-protein-design
        """
        Pos emb.

        Parameters
        ----------
        edge_index : Any
            Input argument.
        num_pos_emb : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def forward(self, batch_data: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the forward pass.

        Parameters
        ----------
        batch_data : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        z, pos, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.batch
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs
        esm_embs = batch_data.esm_emb

        device = z.device

        if self.level == "aminoacid":
            x = self.embedding(z)
        elif self.level == "backbone":
            x = torch.cat(
                [torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs], dim=1
            )
            x = self.embedding(x)
        elif self.level == "allatom":
            x = torch.cat(
                [
                    torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()),
                    bb_embs,
                    side_chain_embs,
                ],
                dim=1,
            )
            x = self.embedding(x)
        elif self.level == "backbone+esm":
            x = torch.cat(
                [torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs, esm_embs],
                dim=1,
            )
            x = self.embedding(x)
        elif self.level == "allatom+esm":
            x = torch.cat(
                [
                    torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()),
                    bb_embs,
                    side_chain_embs,
                ],
                dim=1,
            )
            x = self.embedding(x)
            x_esm = self.embedding_esm(esm_embs)
            x = torch.cat([x, x_esm], dim=-1)
        else:
            print("No supported model!")

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)  # edge position embedding
        j, i = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)

        num_nodes = len(z)

        # Calculate angles theta and phi.
        refi0 = (i - 1) % num_nodes
        refi1 = (i + 1) % num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i], dim=-1).norm(dim=-1)
        theta = torch.atan2(b, a)  # angle of vector (i, i-1) and (i, neighbro)

        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i], dim=-1)
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i], dim=-1)
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2, dim=-1) * (pos[refi0] - pos[i])).sum(dim=-1) / (
            (pos[refi0] - pos[i]).norm(dim=-1)
        )
        phi = torch.atan2(b, a)

        sbf0, feature0 = self.feature0(dist, theta, phi)  # base feature

        ### ***
        if "backbone" in self.level or "allatom" in self.level:
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i], dim=-1), dim=-1)
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7

            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j], dim=-1), dim=-1)
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z, dim=-1)

            angle1 = torch.atan2(
                (torch.cross(Or1_x, Or1_Or2_N, dim=-1) * Or1_z).sum(dim=-1) / Or1_z_length,
                (Or1_x * Or1_Or2_N).sum(dim=-1),
            )
            angle2 = torch.atan2(
                torch.cross(Or1_z, Or2_z, dim=-1).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1)
            )
            angle3 = torch.atan2(
                (torch.cross(Or1_Or2_N, Or2_x, dim=-1) * Or2_z).sum(dim=-1) / Or2_z_length,
                (Or1_Or2_N * Or2_x).sum(dim=-1),
            )

            if self.euler_noise:
                euler_noise = torch.clip(
                    torch.empty(3, len(angle1)).to(device).normal_(mean=0.0, std=0.025),
                    min=-0.1,
                    max=0.1,
                )
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat(
                (
                    self.feature1(dist, angle1),
                    self.feature1(dist, angle2),
                    self.feature1(dist, angle3),
                ),
                1,
            )

        elif self.level == "aminoacid":
            refi = (i - 1) % num_nodes

            refj0 = (j - 1) % num_nodes
            refj = (j - 1) % num_nodes
            refj1 = (j + 1) % num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i], dim=-1)
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j], dim=-1)
            a = (plane1 * plane2).sum(dim=-1)
            b = (torch.cross(plane1, plane2, dim=-1) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau)

        # Interaction blocks.
        # for interaction_block in self.interaction_blocks:
        #     if self.data_augment_eachlayer:
        #         # add gaussian noise to features
        #         gaussian_noise = torch.clip(torch.empty(x.shape).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
        #         x += gaussian_noise
        #     x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)
        # xb = batchgraph2batch(x, batch_data.batch).to(device)
        # posb = batchgraph2batch(pos, batch_data.batch).to(device)
        if True:
            x_e = self.model_E(
                f_in=x,
                pos=pos,
                batch=batch,
                node_atom=x,
                feature0=feature0,
                feature1=feature1,
                pos_emb=pos_emb,
                edge_index=edge_index,
            )
        # xe = self.eqmodel[0](xb, posb)
        # y = scatter(x, batch, dim=0)

        # for lin in self.lins_out:
        #     y = self.relu(lin(y))
        #     y = self.dropout(y)
        # y = self.lin_out(y)
        for lin in self.lins_node_out:
            x = self.relu(lin(x))
            x = self.dropout(x)
        emb = x
        x = self.lin_node_out(x)
        out = F.softmax(x, -1)
        out = out[:, 1]
        if "equi":
            out_e = F.softmax(x_e, -1)
        # out = F.sigmoid(out)
        if True:
            return out_e[:, 1], x_e, emb
        else:
            return out, x, emb

    @property
    def num_params(self) -> int:
        """
        Num params.

        Returns
        -------
        Any
            Computed property value.
        """
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(
        cls,
        *,
        binding_type: str = "DNA",
        model_path: str | Path | None = None,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> EquiSite:
        """Create a model and load checkpoint weights.

        Parameters
        ----------
        binding_type : str
            One of ``"DNA"`` or ``"RNA"``.
        model_path : str | Path | None
            Explicit path to a checkpoint file. When *None*, the default
            checkpoint for ``binding_type`` is used.
        device : str | None
            Device string (e.g. ``"cpu"``, ``"cuda"``). Autodetected when *None*.
        model_kwargs : dict[str, Any] | None
            Extra keyword arguments forwarded to the ``EquiSite`` constructor.

        Returns
        -------
        EquiSite
            Model with loaded weights, placed on ``device`` in eval mode.
        """
        from ._pretrained import resolve_checkpoint_path

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = resolve_checkpoint_path(binding_type, model_path)

        model = cls(**(model_kwargs or {}))
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model


def batchgraph2batch(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Batchgraph2batch.

    Parameters
    ----------
    x : Any
        Input argument.
    batch : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    max_node_num = torch.unique(batch, return_counts=True)[1].max().item()
    batch_size = batch.max().item() + 1
    # Reorganize x by stacking node features per graph using batch indices.
    reshaped_x = torch.zeros(batch_size, max_node_num, x.size(1))  # Initialize the padded tensor.
    for i in range(batch_size):
        # Find indices belonging to the current graph in the batch.
        idx = (batch == i).nonzero().squeeze()
        # Place node features for this graph into the padded tensor.
        reshaped_x[i, : idx.size(0), :] = x[idx]

    return reshaped_x
