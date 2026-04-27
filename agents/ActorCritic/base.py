import torch
import torch.nn as nn
from torch.distributions import Distribution

from omnisafe.typing import Activation, InitFunction


class Actor(nn.Module):
    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:

        nn.Module.__init__(self)
        self._obs_dim: int = obs_space_dim
        self._act_dim: int = act_space_dim
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes
        self._after_inference: bool = False

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        pass

    def forward(self, obs: torch.Tensor) -> Distribution:
        pass

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        pass

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        pass


class Critic(nn.Module):
    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_obs_encoder: bool = False,
    ) -> None:

        nn.Module.__init__(self)
        self._obs_dim: int = obs_space_dim
        self._act_dim: int = act_space_dim
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes
        self._num_critics: int = num_critics
        self._use_obs_encoder: bool = use_obs_encoder
