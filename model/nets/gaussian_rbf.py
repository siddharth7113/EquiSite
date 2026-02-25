"""Gaussian radial basis function layer for edge distances."""

import torch


@torch.jit.script
def gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Gaussian.

    Parameters
    ----------
    x : Any
        Input argument.
    mean : Any
        Input argument.
    std : Any
        Input argument.

    Returns
    -------
    Any
        Function output.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


# From Graphormer
class GaussianRadialBasisLayer(torch.nn.Module):
    """
    GaussianRadialBasisLayer implementation.

    Parameters
    ----------
    num_basis : Any
        Initialization argument.
    cutoff : Any
        Initialization argument.
    """

    def __init__(self, num_basis: int, cutoff: float) -> None:
        """
        Initialize GaussianRadialBasisLayer.

        Parameters
        ----------
        num_basis : Any
            Input argument.
        cutoff : Any
            Input argument.

        """
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = torch.nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1))

        self.std_init_max = 1.0
        self.std_init_min = 1.0 / self.num_basis
        self.mean_init_max = 1.0
        self.mean_init_min = 0
        torch.nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        torch.nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)

    def forward(
        self,
        dist: torch.Tensor,
        node_atom: torch.Tensor | None = None,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the forward pass.

        Parameters
        ----------
        dist : Any
            Input argument.
        node_atom : Any
            Input argument.
        edge_src : Any
            Input argument.
        edge_dst : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x

    def extra_repr(self) -> str:
        """
        Extra repr.

        Returns
        -------
        Any
            Function output.
        """
        return f"mean_init_max={self.mean_init_max}, mean_init_min={self.mean_init_min}, std_init_max={self.std_init_max}, std_init_min={self.std_init_min}"
