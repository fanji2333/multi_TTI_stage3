import torch

from omnisafe.typing import DEVICE_CPU, AdvatageEstimator
from omnisafe.utils import distributed
from omnisafe.utils.math import discount_cumsum

from common.base_buffer import BaseBuffer


class OnPolicyBuffer(BaseBuffer):

    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        cost_dim:int,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = DEVICE_CPU,
    ) -> None:

        super().__init__(obs_space_dim, act_space_dim, cost_dim, size, device)

        self._standardized_adv_r: bool = standardized_adv_r
        self._standardized_adv_c: bool = standardized_adv_c
        self.data['adv_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['discounted_ret'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['target_value_r'] = torch.zeros((size,), dtype=torch.float32, device=device)
        # TODO：修改了cost的维度
        self.data['adv_c'] = torch.zeros((size, cost_dim), dtype=torch.float32, device=device)
        self.data['value_c'] = torch.zeros((size, cost_dim), dtype=torch.float32, device=device)
        self.data['target_value_c'] = torch.zeros((size, cost_dim), dtype=torch.float32, device=device)
        self.data['logp'] = torch.zeros((size,), dtype=torch.float32, device=device)

        self._gamma: float = gamma
        self._lam: float = lam
        self._lam_c: float = lam_c
        self._penalty_coefficient: float = penalty_coefficient
        self._advantage_estimator: AdvatageEstimator = advantage_estimator
        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size: int = size

        assert self._penalty_coefficient >= 0, 'penalty_coefficient must be non-negative!'
        assert self._advantage_estimator in ['gae', 'gae-rtg', 'vtrace', 'plain']

    @property
    def standardized_adv_r(self) -> bool:

        return self._standardized_adv_r

    @property
    def standardized_adv_c(self) -> bool:

        return self._standardized_adv_c

    def store(self, **data: torch.Tensor) -> None:

        assert self.ptr < self.max_size, 'No more space in the buffer!'
        for key, value in data.items():
            self.data[key][self.ptr] = value
        self.ptr += 1

    def finish_path(
        self,
        last_value_r: torch.Tensor = None,
        last_value_c: torch.Tensor = None,
    ) -> None:

        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)

        path_slice = slice(self.path_start_idx, self.ptr)
        # last_value_r = last_value_r.unsqueeze(0).to(self._device)
        last_value_c = last_value_c.unsqueeze(0).to(self._device)
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discountred_ret = discount_cumsum(rewards, self._gamma)[:-1]
        self.data['discounted_ret'][path_slice] = discountred_ret
        # rewards -= self._penalty_coefficient * costs

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:

        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data

    def _calculate_adv_and_value_targets(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self._advantage_estimator == 'gae':

            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, self._gamma * lam)
            target_value = adv + values[:-1]

        elif self._advantage_estimator == 'gae-rtg':

            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            adv = discount_cumsum(deltas, self._gamma * lam)
            # compute rewards-to-go, to be targets for the value function update
            target_value = discount_cumsum(rewards, self._gamma)[:-1]

        elif self._advantage_estimator == 'vtrace':

            path_slice = slice(self.path_start_idx, self.ptr)
            action_probs = self.data['logp'][path_slice].exp()
            target_value, adv, _ = self._calculate_v_trace(
                policy_action_probs=action_probs,
                values=values,
                rewards=rewards,
                behavior_action_probs=action_probs,
                gamma=self._gamma,
                rho_bar=1.0,
                c_bar=1.0,
            )

        elif self._advantage_estimator == 'plain':

            adv = rewards[:-1] + self._gamma * values[1:] - values[:-1]
            target_value = discount_cumsum(rewards, self._gamma)[:-1]

        else:
            raise NotImplementedError

        return adv, target_value

    @staticmethod
    def _calculate_v_trace(
        policy_action_probs: torch.Tensor,
        values: torch.Tensor,  # including bootstrap
        rewards: torch.Tensor,  # including bootstrap
        behavior_action_probs: torch.Tensor,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert values.ndim == 1, 'Please provide arrays instead of scalars'
        assert rewards.ndim == 1, 'Please provide arrays instead of scalars'
        assert policy_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
        assert behavior_action_probs.ndim == 1, 'Please provide arrays instead of scalars'
        assert c_bar <= rho_bar, 'c_bar should be less than or equal to rho_bar'

        sequence_length = policy_action_probs.shape[0]

        rhos = torch.div(policy_action_probs, behavior_action_probs)
        clip_rhos = torch.min(
            rhos,
            torch.as_tensor(rho_bar),
        )
        clip_cs = torch.min(
            rhos,
            torch.as_tensor(c_bar),
        )
        v_s = values[:-1].clone()  # copy all values except bootstrap value
        last_v_s = values[-1]  # bootstrap from last state

        # calculate v_s
        for index in reversed(range(sequence_length)):
            delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
            v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
            last_v_s = v_s[index]  # accumulate current v_s for next iteration

        # calculate q_targets
        v_s_plus_1 = torch.cat((v_s[1:], values[-1:]))
        policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

        return v_s, policy_advantage, clip_rhos
