import numpy as np
import torch

from algo.PPO import PPO
# from omnisafe.common.lagrange import Lagrange
from common.lagrange import Lag


class PPOBaseline(PPO):

    def _init(self) -> None:

        super()._init()
        self._penalty = self._cfgs.penalty_cfgs.penalty

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:

        # 计算加权的成本优势和
        cost_adv_sum = torch.sum(adv_c, dim=1, keepdim=True)

        return (adv_r - cost_adv_sum * self._penalty) / (1 + self._penalty * self._env.get_cost_num())

        # return adv_r - cost_adv_sum * self._penalty
