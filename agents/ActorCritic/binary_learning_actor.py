import torch
import torch.nn as nn
from torch.distributions import Distribution, Bernoulli

from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network

from agents.ActorCritic.base import Actor


class BinaryLearningActor(Actor):

    _current_dist: Bernoulli

    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:

        super().__init__(obs_space_dim, act_space_dim, hidden_sizes, activation, weight_initialization_mode)

        self.mean: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.sigmoid = nn.Sigmoid()
        # self.log_std: nn.Parameter = nn.Parameter(torch.zeros(self._act_dim), requires_grad=True)

    def _distribution(self, obs: torch.Tensor) -> Bernoulli:

        prb = self.sigmoid(self.mean(obs))
        # assert not torch.isnan(obs).any(), "obs distri contains NaN values"
        # assert not torch.isnan(mean).any(), "mean contains NaN values"
        # std = torch.exp(self.log_std)
        # assert not torch.isnan(std).any(), "std contains NaN values"
        # return Normal(mean, std)
        return Bernoulli(probs=prb)

    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:

        self._current_dist = self._distribution(obs)
        self._after_inference = True
        if deterministic:
            # return self._current_dist.mean
            probs = self._current_dist.probs
            return torch.where(probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs))
        return self._current_dist.sample()

    def forward(self, obs: torch.Tensor) -> Distribution:

        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:

        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        return self._current_dist.log_prob(act).sum(axis=-1)

    # @property
    # def std(self) -> float:
    #
    #     return torch.exp(self.log_std).mean().item()
    #
    # @std.setter
    # def std(self, std: float) -> None:
    #     device = self.log_std.device
    #     self.log_std.data.fill_(torch.log(torch.tensor(std, device=device)))
