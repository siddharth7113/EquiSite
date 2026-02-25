"""SE(3)-equivariant graph attention transformer backbone."""

from typing import Any

import torch
import torch.nn.functional as f
import torch_geometric
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

from utils.padding import fea1_sphere_padding, sphere_padding

from .drop import EquivariantDropout, GraphDropPath
from .fast_activation import Activation, Gate
from .fast_layer_norm import EquivariantLayerNormFast
from .gaussian_rbf import GaussianRadialBasisLayer
from .graph_norm import EquivariantGraphNorm
from .instance_norm import EquivariantInstanceNorm
from .layer_norm import EquivariantLayerNormV2

# for bessel radial basis
from .radial_basis import RadialBasis
from .radial_func import RadialProfile
from .registry import register_model
from .tensor_product_rescale import (
    FullyConnectedTensorProductRescale,
    LinearRS,
    TensorProductRescale,
    irreps2gate,
    sort_irreps_even_first,
)

_RESCALE = True
_USE_BIAS = True

# QM9
_MAX_ATOM_TYPE = 5
# Statistics of QM9 with cutoff radius = 5
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


def get_norm_layer(norm_type: Any) -> Any:
    """
    Get norm layer.

    Parameters
    ----------
    norm_type : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    if norm_type == "graph":
        return EquivariantGraphNorm
    elif norm_type == "instance":
        return EquivariantInstanceNorm
    elif norm_type == "layer":
        return EquivariantLayerNormV2
    elif norm_type == "fast_layer":
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError(f"Norm type {norm_type} not supported.")


class SmoothLeakyReLU(torch.nn.Module):
    """
    SmoothLeakyReLU implementation.

    Parameters
    ----------
    negative_slope : Any
        Initialization argument.
    """

    def __init__(self, negative_slope: Any = 0.2) -> Any:
        """
        Initialize SmoothLeakyReLU.

        Parameters
        ----------
        negative_slope : Any
            Input argument.

        """
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x: Any) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self) -> Any:
        """
        Extra repr.

        Returns
        -------
        Any
            Function output.
        """
        return f"negative_slope={self.alpha}"


def get_mul_0(irreps: Any) -> Any:
    """
    Get mul 0.

    Parameters
    ----------
    irreps : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    """
    FullyConnectedTensorProductRescaleNorm implementation.

    Parameters
    ----------
    irreps_in1 : Any
        Initialization argument.
    irreps_in2 : Any
        Initialization argument.
    irreps_out : Any
        Initialization argument.
    bias : Any
        Initialization argument.
    rescale : Any
        Initialization argument.
    internal_weights : Any
        Initialization argument.
    shared_weights : Any
        Initialization argument.
    normalization : Any
        Initialization argument.
    norm_layer : Any
        Initialization argument.
    """

    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        irreps_out: Any,
        bias: Any = True,
        rescale: Any = True,
        internal_weights: Any = None,
        shared_weights: Any = None,
        normalization: Any = None,
        norm_layer: Any = "graph",
    ) -> Any:
        """
        Initialize FullyConnectedTensorProductRescaleNorm.

        Parameters
        ----------
        irreps_in1 : Any
            Input argument.
        irreps_in2 : Any
            Input argument.
        irreps_out : Any
            Input argument.
        bias : Any
            Input argument.
        rescale : Any
            Input argument.
        internal_weights : Any
            Input argument.
        shared_weights : Any
            Input argument.
        normalization : Any
            Input argument.
        norm_layer : Any
            Input argument.

        """
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)

    def forward(self, x: Any, y: Any, batch: Any, weight: Any = None) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.
        y : Any
            Input argument.
        batch : Any
            Input argument.
        weight : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out


class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    """
    FullyConnectedTensorProductRescaleNormSwishGate implementation.

    Parameters
    ----------
    irreps_in1 : Any
        Initialization argument.
    irreps_in2 : Any
        Initialization argument.
    irreps_out : Any
        Initialization argument.
    bias : Any
        Initialization argument.
    rescale : Any
        Initialization argument.
    internal_weights : Any
        Initialization argument.
    shared_weights : Any
        Initialization argument.
    normalization : Any
        Initialization argument.
    norm_layer : Any
        Initialization argument.
    """

    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        irreps_out: Any,
        bias: Any = True,
        rescale: Any = True,
        internal_weights: Any = None,
        shared_weights: Any = None,
        normalization: Any = None,
        norm_layer: Any = "graph",
    ) -> Any:
        """
        Initialize FullyConnectedTensorProductRescaleNormSwishGate.

        Parameters
        ----------
        irreps_in1 : Any
            Input argument.
        irreps_in2 : Any
            Input argument.
        irreps_out : Any
            Input argument.
        bias : Any
            Input argument.
        rescale : Any
            Input argument.
        internal_weights : Any
            Input argument.
        shared_weights : Any
            Input argument.
        normalization : Any
            Input argument.
        norm_layer : Any
            Input argument.

        """
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
            norm_layer=norm_layer,
        )
        self.gate = gate

    def forward(self, x: Any, y: Any, batch: Any, weight: Any = None) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.
        y : Any
            Input argument.
        batch : Any
            Input argument.
        weight : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    """
    FullyConnectedTensorProductRescaleSwishGate implementation.

    Parameters
    ----------
    irreps_in1 : Any
        Initialization argument.
    irreps_in2 : Any
        Initialization argument.
    irreps_out : Any
        Initialization argument.
    bias : Any
        Initialization argument.
    rescale : Any
        Initialization argument.
    internal_weights : Any
        Initialization argument.
    shared_weights : Any
        Initialization argument.
    normalization : Any
        Initialization argument.
    """

    def __init__(
        self,
        irreps_in1: Any,
        irreps_in2: Any,
        irreps_out: Any,
        bias: Any = True,
        rescale: Any = True,
        internal_weights: Any = None,
        shared_weights: Any = None,
        normalization: Any = None,
    ) -> Any:
        """
        Initialize FullyConnectedTensorProductRescaleSwishGate.

        Parameters
        ----------
        irreps_in1 : Any
            Input argument.
        irreps_in2 : Any
            Input argument.
        irreps_out : Any
            Input argument.
        bias : Any
            Input argument.
        rescale : Any
            Input argument.
        internal_weights : Any
            Input argument.
        shared_weights : Any
            Input argument.
        normalization : Any
            Input argument.

        """
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.gate = gate

    def forward(self, x: Any, y: Any, weight: Any = None) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.
        y : Any
            Input argument.
        weight : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out


def DepthwiseTensorProduct(
    irreps_node_input: Any,
    irreps_edge_attr: Any,
    irreps_node_output: Any,
    internal_weights: Any = False,
    bias: Any = True,
) -> Any:
    """
    The irreps of output is pre-determined.
    `irreps_node_output` is used to get certain types of vectors.
    """
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [
        (i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions
    ]
    tp = TensorProductRescale(
        irreps_node_input,
        irreps_edge_attr,
        irreps_output,
        instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
    )
    return tp


class SeparableFCTP(torch.nn.Module):
    """
    Use separable FCTP for spatial convolution.
    """

    def __init__(
        self,
        irreps_node_input: Any,
        irreps_edge_attr: Any,
        irreps_node_output: Any,
        fc_neurons: Any,
        use_activation: Any = False,
        norm_layer: Any = "graph",
        internal_weights: Any = False,
    ) -> Any:
        """
        Initialize SeparableFCTP.

        Parameters
        ----------
        irreps_node_input : Any
            Input argument.
        irreps_edge_attr : Any
            Input argument.
        irreps_node_output : Any
            Input argument.
        fc_neurons : Any
            Input argument.
        use_activation : Any
            Input argument.
        norm_layer : Any
            Input argument.
        internal_weights : Any
            Input argument.

        """
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)

        self.dtp = DepthwiseTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_output,
            bias=False,
            internal_weights=internal_weights,
        )

        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars,
                    [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates,
                    [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated,  # gated tensors
                )
            self.gate = gate

    def forward(
        self, node_input: Any, edge_attr: Any, edge_scalars: Any, batch: Any = None, **kwargs: Any
    ) -> Any:
        """
        Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
        self.dtp_rad(`edge_scalars`).
        """
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [N, irreps_mid] to vectors of shape
    [N, num_heads, irreps_head].
    """

    def __init__(self, irreps_head: Any, num_heads: Any) -> Any:
        """
        Initialize Vec2AttnHeads.

        Parameters
        ----------
        irreps_head : Any
            Input argument.
        num_heads : Any
            Input argument.

        """
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x: Any) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self) -> Any:
        """
        Return a readable string representation.

        Returns
        -------
        Any
            Function output.
        """
        return (
            f"{self.__class__.__name__}(irreps_head={self.irreps_head}, num_heads={self.num_heads})"
        )


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [N, num_heads, irreps_head] into
    vectors of shape [N, irreps_head * num_heads].
    """

    def __init__(self, irreps_head: Any) -> Any:
        """
        Initialize AttnHeads2Vec.

        Parameters
        ----------
        irreps_head : Any
            Input argument.

        """
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x: Any) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self) -> Any:
        """
        Return a readable string representation.

        Returns
        -------
        Any
            Function output.
        """
        return f"{self.__class__.__name__}(irreps_head={self.irreps_head})"


class ConcatIrrepsTensor(torch.nn.Module):
    """
    ConcatIrrepsTensor implementation.

    Parameters
    ----------
    irreps_1 : Any
        Initialization argument.
    irreps_2 : Any
        Initialization argument.
    """

    def __init__(self, irreps_1: Any, irreps_2: Any) -> Any:
        """
        Initialize ConcatIrrepsTensor.

        Parameters
        ----------
        irreps_1 : Any
            Input argument.
        irreps_2 : Any
            Input argument.

        """
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)

        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out)  # self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()

        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i, 1)))
        irreps_max = o3.Irreps(irreps_max)

        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2

    def get_irreps_dim(self, irreps: Any) -> Any:
        """
        Get irreps dim.

        Parameters
        ----------
        irreps : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls

    def check_sorted(self, irreps: Any) -> Any:
        """
        Check sorted.

        Parameters
        ----------
        irreps : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, f"Parity order error: {irreps}"
            assert lmax <= ir.l

    def get_ir_index(self, ir: Any, irreps: Any) -> Any:
        """
        Get ir index.

        Parameters
        ----------
        ir : Any
            Input argument.
        irreps : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1

    def forward(self, feature_1: Any, feature_2: Any) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        feature_1 : Any
            Input argument.
        feature_2 : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output

    def __repr__(self) -> Any:
        """
        Return a readable string representation.

        Returns
        -------
        Any
            Function output.
        """
        return f"{self.__class__.__name__}(irreps_1={self.irreps_1}, irreps_2={self.irreps_2})"


@compile_mode("script")
class GraphAttention(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)
    """

    def __init__(
        self,
        irreps_node_input: Any,
        irreps_node_attr: Any,
        irreps_edge_attr: Any,
        irreps_node_output: Any,
        fc_neurons: Any,
        irreps_head: Any,
        num_heads: Any,
        irreps_pre_attn: Any = None,
        rescale_degree: Any = False,
        nonlinear_message: Any = False,
        alpha_drop: Any = 0.1,
        proj_drop: Any = 0.1,
    ) -> Any:
        """
        Initialize GraphAttention.

        Parameters
        ----------
        irreps_node_input : Any
            Input argument.
        irreps_node_attr : Any
            Input argument.
        irreps_edge_attr : Any
            Input argument.
        irreps_node_output : Any
            Input argument.
        fc_neurons : Any
            Input argument.
        irreps_head : Any
            Input argument.
        num_heads : Any
            Input argument.
        irreps_pre_attn : Any
            Input argument.
        rescale_degree : Any
            Input argument.
        nonlinear_message : Any
            Input argument.
        alpha_drop : Any
            Input argument.
        proj_drop : Any
            Input argument.

        """
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)

        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(
            irreps_attn_heads
        )  # irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps(f"{mul_alpha}x0e")  # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                self.irreps_pre_attn,
                fc_neurons,
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_heads,
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=True,
            )
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps(f"{mul_alpha_head}x0e"), num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_all,
                fc_neurons,
                use_activation=False,
                norm_layer=None,
            )
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps(f"{mul_alpha_head}x0e") + irreps_head).simplify(), num_heads
            )

        self.alpha_act = Activation(o3.Irreps(f"{mul_alpha_head}x0e"), [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot)  # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, drop_prob=proj_drop)

    def forward(
        self,
        node_input: Any,
        node_attr: Any,
        edge_src: Any,
        edge_dst: Any,
        edge_attr: Any,
        edge_scalars: Any,
        batch: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        node_input : Any
            Input argument.
        node_attr : Any
            Input argument.
        edge_src : Any
            Input argument.
        edge_dst : Any
            Input argument.
        edge_attr : Any
            Input argument.
        edge_scalars : Any
            Input argument.
        batch : Any
            Input argument.
        **kwargs : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        fea0 = True
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        if fea0:
            message = message + edge_attr[1] + edge_attr[2] + edge_attr[3]

            # edge_attr[0] = edge_attr[0] + edge_attr[1]
        # message_o = message
        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr[0], weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr[0], edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))

        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum("bik, aik -> bi", alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(
                edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype
            )
            degree = degree.view(-1, 1)
            attn = attn * degree

        node_output = self.proj(attn)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

    def extra_repr(self) -> Any:
        """
        Extra repr.

        Returns
        -------
        Any
            Function output.
        """
        output_str = super().extra_repr()
        output_str = output_str + f"rescale_degree={self.rescale_degree}, "
        return output_str


@compile_mode("script")
class FeedForwardNetwork(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """

    def __init__(
        self,
        irreps_node_input: Any,
        irreps_node_attr: Any,
        irreps_node_output: Any,
        irreps_mlp_mid: Any = None,
        proj_drop: Any = 0.1,
    ) -> Any:
        """
        Initialize FeedForwardNetwork.

        Parameters
        ----------
        irreps_node_input : Any
            Input argument.
        irreps_node_attr : Any
            Input argument.
        irreps_node_output : Any
            Input argument.
        irreps_mlp_mid : Any
            Input argument.
        proj_drop : Any
            Input argument.

        """
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None else self.irreps_node_input
        )
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_mlp_mid,
            bias=True,
            rescale=_RESCALE,
        )
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid,
            self.irreps_node_attr,
            self.irreps_node_output,
            bias=True,
            rescale=_RESCALE,
        )

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, drop_prob=proj_drop)

    def forward(self, node_input: Any, node_attr: Any, **kwargs: Any) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        node_input : Any
            Input argument.
        node_attr : Any
            Input argument.
        **kwargs : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


@compile_mode("script")
class TransBlock(torch.nn.Module):
    """
    1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    """

    def __init__(
        self,
        irreps_node_input: Any,
        irreps_node_attr: Any,
        irreps_edge_attr: Any,
        irreps_node_output: Any,
        fc_neurons: Any,
        irreps_head: Any,
        num_heads: Any,
        irreps_pre_attn: Any = None,
        rescale_degree: Any = False,
        nonlinear_message: Any = False,
        alpha_drop: Any = 0.1,
        proj_drop: Any = 0.1,
        drop_path_rate: Any = 0.0,
        irreps_mlp_mid: Any = None,
        norm_layer: Any = "layer",
    ) -> Any:
        """
        Initialize TransBlock.

        Parameters
        ----------
        irreps_node_input : Any
            Input argument.
        irreps_node_attr : Any
            Input argument.
        irreps_edge_attr : Any
            Input argument.
        irreps_node_output : Any
            Input argument.
        fc_neurons : Any
            Input argument.
        irreps_head : Any
            Input argument.
        num_heads : Any
            Input argument.
        irreps_pre_attn : Any
            Input argument.
        rescale_degree : Any
            Input argument.
        nonlinear_message : Any
            Input argument.
        alpha_drop : Any
            Input argument.
        proj_drop : Any
            Input argument.
        drop_path_rate : Any
            Input argument.
        irreps_mlp_mid : Any
            Input argument.
        norm_layer : Any
            Input argument.

        """
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None else self.irreps_node_input
        )

        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=self.rescale_degree,
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        # self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input,
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop,
        )
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input,
                self.irreps_node_attr,
                self.irreps_node_output,
                bias=True,
                rescale=_RESCALE,
            )

    def forward(
        self,
        node_input: Any,
        node_attr: Any,
        edge_src: Any,
        edge_dst: Any,
        edge_attr: Any,
        edge_scalars: Any,
        batch: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        node_input : Any
            Input argument.
        node_attr : Any
            Input argument.
        edge_src : Any
            Input argument.
        edge_dst : Any
            Input argument.
        edge_attr : Any
            Input argument.
        edge_scalars : Any
            Input argument.
        batch : Any
            Input argument.
        **kwargs : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        # norm_1_output = node_features
        node_features = self.ga(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        )

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        # node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        return node_output


class NodeEmbeddingNetwork(torch.nn.Module):
    """
    NodeEmbeddingNetwork implementation.

    Parameters
    ----------
    irreps_node_embedding : Any
        Initialization argument.
    max_atom_type : Any
        Initialization argument.
    bias : Any
        Initialization argument.
    """

    def __init__(
        self, irreps_node_embedding: Any, max_atom_type: Any = _MAX_ATOM_TYPE, bias: Any = True
    ) -> Any:
        """
        Initialize NodeEmbeddingNetwork.

        Parameters
        ----------
        irreps_node_embedding : Any
            Input argument.
        max_atom_type : Any
            Input argument.
        bias : Any
            Input argument.

        """
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(
            o3.Irreps(f"{self.max_atom_type}x0e"), self.irreps_node_embedding, bias=bias
        )
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type**0.5)

    def forward(self, node_atom: Any) -> Any:
        """
        `node_atom` is a LongTensor.
        """
        node_atom_onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).float()
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)

        return node_embedding, node_attr, node_atom_onehot


class ScaledScatter(torch.nn.Module):
    """
    ScaledScatter implementation.

    Parameters
    ----------
    avg_aggregate_num : Any
        Initialization argument.
    """

    def __init__(self, avg_aggregate_num: Any) -> Any:
        """
        Initialize ScaledScatter.

        Parameters
        ----------
        avg_aggregate_num : Any
            Input argument.

        """
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0

    def forward(self, x: Any, index: Any, **kwargs: Any) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        x : Any
            Input argument.
        index : Any
            Input argument.
        **kwargs : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num**0.5)
        return out

    def extra_repr(self) -> Any:
        """
        Extra repr.

        Returns
        -------
        Any
            Function output.
        """
        return f"avg_aggregate_num={self.avg_aggregate_num}"


class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    """
    EdgeDegreeEmbeddingNetwork implementation.

    Parameters
    ----------
    irreps_node_embedding : Any
        Initialization argument.
    irreps_edge_attr : Any
        Initialization argument.
    fc_neurons : Any
        Initialization argument.
    avg_aggregate_num : Any
        Initialization argument.
    """

    def __init__(
        self,
        irreps_node_embedding: Any,
        irreps_edge_attr: Any,
        fc_neurons: Any,
        avg_aggregate_num: Any,
    ) -> Any:
        """
        Initialize EdgeDegreeEmbeddingNetwork.

        Parameters
        ----------
        irreps_node_embedding : Any
            Input argument.
        irreps_edge_attr : Any
            Input argument.
        fc_neurons : Any
            Input argument.
        avg_aggregate_num : Any
            Input argument.

        """
        super().__init__()
        self.exp = LinearRS(
            o3.Irreps("1x0e"), irreps_node_embedding, bias=_USE_BIAS, rescale=_RESCALE
        )
        self.dw = DepthwiseTensorProduct(
            irreps_node_embedding,
            irreps_edge_attr,
            irreps_node_embedding,
            internal_weights=False,
            bias=False,
        )
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for slice, slice_sqrt_k in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(
        self,
        node_input: Any,
        edge_attr: Any,
        edge_scalars: Any,
        edge_src: Any,
        edge_dst: Any,
        batch: Any,
    ) -> Any:
        """
        Run the forward pass.

        Parameters
        ----------
        node_input : Any
            Input argument.
        edge_attr : Any
            Input argument.
        edge_scalars : Any
            Input argument.
        edge_src : Any
            Input argument.
        edge_dst : Any
            Input argument.
        batch : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(
            edge_features, edge_dst, dim=0, dim_size=node_features.shape[0]
        )
        return node_features


class GraphAttentionTransformer(torch.nn.Module):
    """
    GraphAttentionTransformer implementation.

    Parameters
    ----------
    irreps_in : Any
        Initialization argument.
    irreps_node_embedding : Any
        Initialization argument.
    num_layers : Any
        Initialization argument.
    irreps_node_attr : Any
        Initialization argument.
    irreps_sh : Any
        Initialization argument.
    max_radius : Any
        Initialization argument.
    number_of_basis : Any
        Initialization argument.
    basis_type : Any
        Initialization argument.
    fc_neurons : Any
        Initialization argument.
    irreps_feature : Any
        Initialization argument.
    irreps_head : Any
        Initialization argument.
    num_heads : Any
        Initialization argument.
    irreps_pre_attn : Any
        Initialization argument.
    rescale_degree : Any
        Initialization argument.
    nonlinear_message : Any
        Initialization argument.
    irreps_mlp_mid : Any
        Initialization argument.
    norm_layer : Any
        Initialization argument.
    alpha_drop : Any
        Initialization argument.
    proj_drop : Any
        Initialization argument.
    out_drop : Any
        Initialization argument.
    drop_path_rate : Any
        Initialization argument.
    mean : Any
        Initialization argument.
    std : Any
        Initialization argument.
    scale : Any
        Initialization argument.
    atomref : Any
        Initialization argument.
    """

    def __init__(
        self,
        irreps_in: Any = "5x0e",
        irreps_node_embedding: Any = "128x0e+64x1e+32x2e",
        num_layers: Any = 6,
        irreps_node_attr: Any = "1x0e",
        irreps_sh: Any = "1x0e+1x1e+1x2e",
        max_radius: Any = 5.0,
        number_of_basis: Any = 128,
        basis_type: Any = "gaussian",
        fc_neurons: Any = [64, 64],
        irreps_feature: Any = "512x0e",
        irreps_head: Any = "32x0e+16x1o+8x2e",
        num_heads: Any = 4,
        irreps_pre_attn: Any = None,
        rescale_degree: Any = False,
        nonlinear_message: Any = False,
        irreps_mlp_mid: Any = "128x0e+64x1e+32x2e",
        norm_layer: Any = "layer",
        alpha_drop: Any = 0.2,
        proj_drop: Any = 0.0,
        out_drop: Any = 0.0,
        drop_path_rate: Any = 0.0,
        mean: Any = None,
        std: Any = None,
        scale: Any = None,
        atomref: Any = None,
    ) -> Any:
        """
        Initialize GraphAttentionTransformer.

        Parameters
        ----------
        irreps_in : Any
            Input argument.
        irreps_node_embedding : Any
            Input argument.
        num_layers : Any
            Input argument.
        irreps_node_attr : Any
            Input argument.
        irreps_sh : Any
            Input argument.
        max_radius : Any
            Input argument.
        number_of_basis : Any
            Input argument.
        basis_type : Any
            Input argument.
        fc_neurons : Any
            Input argument.
        irreps_feature : Any
            Input argument.
        irreps_head : Any
            Input argument.
        num_heads : Any
            Input argument.
        irreps_pre_attn : Any
            Input argument.
        rescale_degree : Any
            Input argument.
        nonlinear_message : Any
            Input argument.
        irreps_mlp_mid : Any
            Input argument.
        norm_layer : Any
            Input argument.
        alpha_drop : Any
            Input argument.
        proj_drop : Any
            Input argument.
        out_drop : Any
            Input argument.
        drop_path_rate : Any
            Input argument.
        mean : Any
            Input argument.
        std : Any
            Input argument.
        scale : Any
            Input argument.
        atomref : Any
            Input argument.

        """
        super().__init__()

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
        self.scale = scale
        self.register_buffer("atomref", atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh)
            if irreps_sh is not None
            else o3.Irreps.spherical_harmonics(self.lmax)
        )
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.basis_type = basis_type
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis, cutoff=self.max_radius, rbf={"name": "spherical_bessel"}
            )
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding, self.irreps_edge_attr, self.fc_neurons, _AVG_DEGREE
        )

        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps("2x0e"), rescale=_RESCALE),
        )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)

    def build_blocks(self) -> Any:
        """
        Build blocks.

        Returns
        -------
        Any
            Function output.
        """
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_block_output = self.irreps_feature
            blk = TransBlock(
                irreps_node_input=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
            )
            self.blocks.append(blk)

    def _init_weights(self, m: Any) -> Any:
        """
         init weights.

        Parameters
        ----------
        m : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Any:
        """
        No weight decay.

        Returns
        -------
        Any
            Function output.
        """
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)
                or isinstance(module, RadialBasis)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and "weight" in parameter_name:
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    def forward(
        self,
        f_in: Any,
        pos: Any,
        batch: Any,
        node_atom: Any,
        feature0: Any,
        feature1: Any,
        pos_emb: Any,
        edge_index: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        # edge_src_t, edge_dst_t = radius_graph(pos, r=self.max_radius, batch=batch,
        #     max_num_neighbors=32)
        """
        Run the forward pass.

        Parameters
        ----------
        f_in : Any
            Input argument.
        pos : Any
            Input argument.
        batch : Any
            Input argument.
        node_atom : Any
            Input argument.
        feature0 : Any
            Input argument.
        feature1 : Any
            Input argument.
        pos_emb : Any
            Input argument.
        edge_index : Any
            Input argument.
        **kwargs : Any
            Input argument.

        Returns
        -------
        torch.Tensor
            Function output.
        """
        edge_src, edge_dst = edge_index[0], edge_index[1]

        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        # edge_vec1 = torch.cat((edge_vec, feature0, feature1, pos_emb), dim=-1)
        feature0 = sphere_padding(feature0, [128, 64, 32], 3)
        feature1 = fea1_sphere_padding(feature1, [128, 64, 32], 3)
        pos_emb = f.pad(pos_emb, (0, 480 - pos_emb.size(-1)), "constant", 0)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization="component"
        )
        edge_sh_list = [edge_sh, feature0, feature1, pos_emb]
        # node_atom = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        # atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)

        atom_embedding = f.pad(f_in, (0, 480 - f_in.size(-1)))
        edge_length = edge_vec.norm(dim=1)
        # edge_length_embedding = sin_pos_embedding(x=edge_length,
        #    start=0.0, end=self.max_radius, number=self.number_of_basis,
        #    cutoff=False)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        for blk in self.blocks:
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh_list,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )

        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        # outputs = self.scale_scatter(outputs, batch, dim=0)

        if self.scale is not None:
            outputs = self.scale * outputs

        return outputs


@register_model
def graph_attention_transformer_l2(
    irreps_in: Any,
    radius: Any,
    num_basis: Any = 128,
    atomref: Any = None,
    task_mean: Any = None,
    task_std: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Graph attention transformer l2.

    Parameters
    ----------
    irreps_in : Any
        Input argument.
    radius : Any
        Input argument.
    num_basis : Any
        Input argument.
    atomref : Any
        Input argument.
    task_mean : Any
        Input argument.
    task_std : Any
        Input argument.
    **kwargs : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="384x0e+192x1e+96x2e",
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model


@register_model
def graph_attention_transformer_nonlinear_l2(
    irreps_in: Any,
    radius: Any,
    num_basis: Any = 128,
    atomref: Any = None,
    task_mean: Any = None,
    task_std: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Graph attention transformer nonlinear l2.

    Parameters
    ----------
    irreps_in : Any
        Input argument.
    radius : Any
        Input argument.
    num_basis : Any
        Input argument.
    atomref : Any
        Input argument.
    task_mean : Any
        Input argument.
    task_std : Any
        Input argument.
    **kwargs : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
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
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model


@register_model
def graph_attention_transformer_nonlinear_l2_e3(
    irreps_in: Any,
    radius: Any,
    num_basis: Any = 128,
    atomref: Any = None,
    task_mean: Any = None,
    task_std: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Graph attention transformer nonlinear l2 e3.

    Parameters
    ----------
    irreps_in : Any
        Input argument.
    radius : Any
        Input argument.
    num_basis : Any
        Input argument.
    atomref : Any
        Input argument.
    task_mean : Any
        Input argument.
    task_std : Any
        Input argument.
    **kwargs : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+32x0o+32x1e+32x1o+16x2e+16x2o",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1o+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        irreps_feature="512x0e",
        irreps_head="32x0e+8x0o+8x1e+8x1o+4x2e+4x2o",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=True,
        irreps_mlp_mid="384x0e+96x0o+96x1e+96x1o+48x2e+48x2o",
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.2
@register_model
def graph_attention_transformer_nonlinear_bessel_l2(
    irreps_in: Any,
    radius: Any,
    num_basis: Any = 128,
    atomref: Any = None,
    task_mean: Any = None,
    task_std: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Graph attention transformer nonlinear bessel l2.

    Parameters
    ----------
    irreps_in : Any
        Input argument.
    radius : Any
        Input argument.
    num_basis : Any
        Input argument.
    atomref : Any
        Input argument.
    task_mean : Any
        Input argument.
    task_std : Any
        Input argument.
    **kwargs : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="bessel",
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
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.1
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop01(
    irreps_in: Any,
    radius: Any,
    num_basis: Any = 128,
    atomref: Any = None,
    task_mean: Any = None,
    task_std: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Graph attention transformer nonlinear bessel l2 drop01.

    Parameters
    ----------
    irreps_in : Any
        Input argument.
    radius : Any
        Input argument.
    num_basis : Any
        Input argument.
    atomref : Any
        Input argument.
    task_mean : Any
        Input argument.
    task_std : Any
        Input argument.
    **kwargs : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="bessel",
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=True,
        irreps_mlp_mid="384x0e+192x1e+96x2e",
        norm_layer="layer",
        alpha_drop=0.1,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.0
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop00(
    irreps_in: Any,
    radius: Any,
    num_basis: Any = 128,
    atomref: Any = None,
    task_mean: Any = None,
    task_std: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Graph attention transformer nonlinear bessel l2 drop00.

    Parameters
    ----------
    irreps_in : Any
        Input argument.
    radius : Any
        Input argument.
    num_basis : Any
        Input argument.
    atomref : Any
        Input argument.
    task_mean : Any
        Input argument.
    task_std : Any
        Input argument.
    **kwargs : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    model = GraphAttentionTransformer(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=[64, 64],
        basis_type="bessel",
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1e+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=True,
        irreps_mlp_mid="384x0e+192x1e+96x2e",
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
    )
    return model
