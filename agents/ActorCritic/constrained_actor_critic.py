import torch
from torch import optim
from omnisafe.utils.config import ModelConfig
from torch.optim.lr_scheduler import ConstantLR, LinearLR

import agents.ActorCritic.actor_critic as ac


class ConstraintActorCritic(ac.ActorCritic):

    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        model_cfgs: ModelConfig,
        epochs: int,
        cost_num:int,
        device
    ) -> None:

        super().__init__(obs_space_dim, act_space_dim, model_cfgs, epochs, device)
        self.cost_critic: ac.Critic = ac.VCritic(
            obs_space_dim=obs_space_dim,
            act_space_dim=act_space_dim,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=cost_num,
        )
        self.add_module('cost_critic', self.cost_critic)

        if model_cfgs.critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )
            if model_cfgs.linear_lr_decay:
                self.cost_critic_scheduler = LinearLR(
                    self.cost_critic_optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=epochs,
                    verbose=True,
                )

    def step(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:

        with torch.no_grad():
            value_r = self.reward_critic(obs)
            value_c = self.cost_critic(obs)

            action = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, torch.tensor(value_r, device=self.device), torch.tensor(value_c, device=self.device), log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:

        return self.step(obs, deterministic=deterministic)
