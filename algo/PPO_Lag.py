import numpy as np
import torch

from algo.PPO import PPO
# from omnisafe.common.lagrange import Lagrange
from common.lagrange import Lag


class PPOLag(PPO):

    def _init(self) -> None:

        super()._init()
        self._lag_num = self._env.get_cost_num()
        self._lagrange: list[Lag] = [Lag(**self._cfgs.lagrange_cfgs) for _ in range(self._lag_num)]
        self._penalty = torch.zeros(self._lag_num).to(self._device)

    def _update(self, epoch) -> None:

        # Jc = self._logger.get_stats('Metrics/EpCost')[0]
        Jc = self._metrics['EpCost'][epoch].cpu() / self._metrics['EpLen'][epoch].cpu()
        # Jc = self._metrics['EpCost'][epoch].cpu()
        # eplen = self._metrics['EpLen'][epoch].cpu()
        # Jc = self._metrics['EpMaxCost'][epoch].cpu()
        for item in Jc:
            assert not np.isnan(item), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        for idx in range(self._lag_num):
            self._lagrange[idx].update_lagrange_multiplier(Jc[idx])
            # 根据epoch总cost来决定是否给予lag乘子
            # if Jc[idx]/eplen > 0:
            #     self._penalty[idx] = self._lagrange[idx].lagrangian_multiplier.item()
            # else:
            #     self._penalty[idx] = 0
            self._penalty[idx] = self._lagrange[idx].lagrangian_multiplier.item()

        print(f'penalty in epoch {epoch + 1} is {self._penalty.tolist()}')

        # then update the policy and value function
        super()._update(epoch)

        self._logger[epoch]['penalty'] = self._penalty.tolist()
        self._logger[epoch]['lag'] = [self._lagrange[idx].lagrangian_multiplier.item() for idx in range(self._lag_num)]

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:

        penalty = self._penalty
        # for idx in range(self._lag_num):
            # penalty[idx] = self._lagrange[idx].lagrangian_multiplier.item()

        # 计算加权的成本优势和
        weighted_cost_adv_sum = torch.sum(adv_c * penalty.unsqueeze(0), dim=1, keepdim=True)

        return (adv_r - weighted_cost_adv_sum) / (1 + torch.sum(penalty))

        # return adv_r - weighted_cost_adv_sum
