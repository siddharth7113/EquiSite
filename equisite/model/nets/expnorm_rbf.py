"""Exponential-normal radial basis function layers."""

import math

import torch


class CosineCutoff(torch.nn.Module):
    """
    CosineCutoff implementation.

    Parameters
    ----------
    cutoff_lower : Any
        Initialization argument.
    cutoff_upper : Any
        Initialization argument.
    """

    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0) -> None:
        """
        Initialize CosineCutoff.

        Parameters
        ----------
        cutoff_lower : Any
            Input argument.
        cutoff_upper : Any
            Input argument.

        """
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Parameters
        ----------
        distances : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py#L111
class ExpNormalSmearing(torch.nn.Module):
    """
    ExpNormalSmearing implementation.

    Parameters
    ----------
    cutoff_lower : Any
        Initialization argument.
    cutoff_upper : Any
        Initialization argument.
    num_rbf : Any
        Initialization argument.
    trainable : Any
        Initialization argument.
    """

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
    ) -> None:
        """
        Initialize ExpNormalSmearing.

        Parameters
        ----------
        cutoff_lower : Any
            Input argument.
        cutoff_upper : Any
            Input argument.
        num_rbf : Any
            Input argument.
        trainable : Any
            Input argument.

        """
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        """
         initial params.

        Returns
        -------
        Any
            Function output.
        """
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self) -> None:
        """
        Reset parameters.

        Returns
        -------
        Any
            Function output.
        """
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Parameters
        ----------
        dist : Any
            Input argument.

        Returns
        -------
        Any
            Function output.
        """
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )
