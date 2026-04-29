import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.schedule import PiecewiseSchedule, Schedule

from agents.ActorCritic.gaussian_learning_actor import GaussianLearningActor
from agents.ActorCritic.binary_learning_actor import BinaryLearningActor
from agents.ActorCritic.Critic.v_critic import VCritic
from agents.ActorCritic.base import Actor, Critic


class ActorCritic(nn.Module):

    std_schedule: Schedule

    def __init__(
        self,
        obs_space_dim: int,
        act_space_dim: int,
        model_cfgs: ModelConfig,
        epochs: int,
        device
    ) -> None:

        super().__init__()

        self.device = device

        if model_cfgs.actor_type == 'gaussian_learning':
            self.actor: Actor = GaussianLearningActor(
                obs_space_dim=obs_space_dim,
                act_space_dim=act_space_dim,
                hidden_sizes=model_cfgs.actor.hidden_sizes,
                activation=model_cfgs.actor.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode
            )
        if model_cfgs.actor_type == 'binary_learning':
            self.actor: Actor = BinaryLearningActor(
                obs_space_dim=obs_space_dim,
                act_space_dim=act_space_dim,
                hidden_sizes=model_cfgs.actor.hidden_sizes,
                activation=model_cfgs.actor.activation,
                weight_initialization_mode=model_cfgs.weight_initialization_mode
            )
        self.reward_critic: Critic = VCritic(
            obs_space_dim=obs_space_dim,
            act_space_dim=act_space_dim,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1
        )
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)

        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: optim.Optimizer
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
            # self.actor_scheduler: LinearLR | ConstantLR
            if model_cfgs.linear_lr_decay:
                self.actor_scheduler = LinearLR(
                    self.actor_optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=epochs,
                    verbose=True,
                )
            else:
                self.actor_scheduler = ConstantLR(
                    self.actor_optimizer,
                    factor=1.0,
                    total_iters=epochs,
                    verbose=True,
                )
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: optim.Optimizer = optim.Adam(
                self.reward_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )
            if model_cfgs.linear_lr_decay:
                self.reward_critic_scheduler = LinearLR(
                    self.reward_critic_optimizer,
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
            act = self.actor.predict(obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(act)
        return act, value_r[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:

        return self.step(obs, deterministic=deterministic)

    def set_annealing(self, epochs: list[int], std: list[float]) -> None:

        assert isinstance(
            self.actor,
            GaussianLearningActor,
        ), 'Only GaussianLearningActor support annealing.'
        self.std_schedule = PiecewiseSchedule(
            endpoints=list(zip(epochs, std)),
            outside_value=std[-1],
        )

    def annealing(self, epoch: int) -> None:

        assert isinstance(
            self.actor,
            GaussianLearningActor,
        ), 'Only GaussianLearningActor support annealing.'
        self.actor.std = self.std_schedule.value(epoch)
