import torch
import torch.nn as nn

from agents.ActorCritic.base import Critic
from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import build_mlp_network

from utils.model import initialize_layer


class VCritic(Critic):

    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
    ) -> None:

        super().__init__(
            obs_space_dim,
            act_space_dim,
            hidden_sizes,
            activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        # self.net_lst: list[nn.Module]
        # self.net_lst = []
        #
        # for idx in range(self._num_critics):
        #     net = build_mlp_network(
        #         # TODO：是否需要改动输出层？还是设置多个critic？
        #         sizes=[self._obs_dim, *self._hidden_sizes, 1],
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #     )
        #     self.net_lst.append(net)
        #     self.add_module(f'critic_{idx}', net)

        # TODO:改为共享底层网络、但独立输出头的critic结构
        # 共享特征提取层
        self.shared_layers = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes],
            activation=self._activation,
            weight_initialization_mode=self._weight_initialization_mode,
        )
        # 独立输出头
        self.heads = nn.ModuleList([
            nn.Linear(self._hidden_sizes[-1], 1) for _ in range(self._num_critics)
        ])
        for head in self.heads:
            # Critic 的输出层（预测单个 value 值），标准做法是 gain=1.0
            initialize_layer(
                init_function=self._weight_initialization_mode,
                layer=head,
                gain=1.0
            )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> list[torch.Tensor]:

        # res = []
        # for critic in self.net_lst:
        #     res.append(torch.squeeze(critic(obs), -1))
        # return res

        features = self.shared_layers(obs)
        return [head(features).squeeze(-1) for head in self.heads]
