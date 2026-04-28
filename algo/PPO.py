import torch

from algo.policygradient import PolicyGradient


class PPO(PolicyGradient):

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        assert not torch.isnan(obs).any(), "obs contains NaN values"
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        # std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        ra = ratio * adv
        rca = ratio_cliped * adv
        loss = -torch.min(ra, rca)
        loss = loss.mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        # self._logger[epoch]['Train/Entropy'].append(entropy)
        # self._logger[epoch]['Train/PolicyRatio'].append(ratio.tolist())
        # self._logger[epoch]['Train/PolicyStd'].append(std)
        # self._logger[epoch]['Loss/Loss_pi'].append(loss.mean().item())
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio.mean().item(),
                # 'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

        return loss