import torch
import torch.nn as nn
import time
import os
import json
from rich.progress import track
from typing import Any
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from omnisafe.utils.tools import get_device
from omnisafe.utils.config import Config
# from omnisafe.common.logger import Logger

from agents.ActorCritic.constrained_actor_critic import ConstraintActorCritic
from common.onpolicy_buffer import OnPolicyBuffer
from common.logger import MyLogger as Logger
# from env import environment as env
# from env import environment_new_distributed as env
from env import environment_QuaDRiGa_SU as env

class PolicyGradient:

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor
    _ep_max_cost: torch.Tensor

    def __init__(self, cfgs: Config, chp: str = None) -> None:

        self._cfgs: Config = cfgs

        # 固定种子
        torch.manual_seed(self._cfgs.seed)
        torch.cuda.manual_seed_all(self._cfgs.seed)

        assert hasattr(cfgs.train_cfgs, 'device'), 'Please specify the device in the config file.'
        self._device: torch.device = get_device(self._cfgs.train_cfgs.device)
        # # 查看设备类型及硬件信息
        # if self._device.type == 'cuda':
        #     # 获取GPU型号
        #     gpu_name = torch.cuda.get_device_name(self._device)
        #     print(f"使用的GPU型号：{gpu_name}")
        #     # 还可以获取更多信息，如GPU内存
        #     total_memory = torch.cuda.get_device_properties(self._device).total_memory / 1024 ** 3  # 转换为GB
        #     print(f"GPU总内存：{total_memory:.2f} GB")
        # else:
        #     print("使用的是CPU，无GPU信息")
        # if torch.cuda.is_available():
        #     print(f"可用GPU数量：{torch.cuda.device_count()}")
        #     for i in range(torch.cuda.device_count()):
        #         print(f"GPU {i}：{torch.cuda.get_device_name(i)}")
        # else:
        #     print("无可用GPU")

        self._ret = []
        self._cost = []
        self._len = []

        self._init_env()
        self._cost_num = self._env.get_cost_num()
        self._init_model(chp)
        self._init()
        self._init_log()

        # self._logger = []
        # self._metrics = {
        #     'EpRet': [],
        #     'EpCost': [],
        #     'EpLen': [],
        #     'EpMaxCost':[]
        # }

        # self._init_log()

    def _init_env(self) -> None:

        self._env: env.Environment = env.Environment(
            self._cfgs.env_cfgs,
            self._device
        )

    def _init_model(self, chp: str = None) -> None:

        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space_dim=self._env.get_obs_dim(),
            act_space_dim=self._env.get_act_dim(),
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
            cost_num=self._cost_num,
            device=self._device
        ).to(self._device)

        if chp:
            model_params = torch.load(chp)
            self._actor_critic.actor.load_state_dict(model_params['actor_param'])
            self._actor_critic.reward_critic.load_state_dict(model_params['r_critic_param'])
            self._actor_critic.cost_critic.load_state_dict(model_params['c_critic_param'])

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:

        self._buf: OnPolicyBuffer = OnPolicyBuffer(
            obs_space_dim=self._env.get_obs_dim(),
            act_space_dim=self._env.get_act_dim(),
            cost_dim=self._env.get_cost_num(),
            size=self._cfgs.algo_cfgs.steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            device=self._device,
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['actor_param'] = self._actor_critic.actor
        what_to_save['r_critic_param'] = self._actor_critic.reward_critic
        what_to_save['c_critic_param'] = self._actor_critic.cost_critic
        if self._cfgs.env_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        if self._cfgs.env_cfgs.reward_normalize:
            reward_normalizer = self._env.save()['reward_normalizer']
            what_to_save['reward_normalizer'] = reward_normalizer
        if self._cfgs.env_cfgs.cost_normalize:
            cost_normalizer = self._env.save()['cost_normalizer']
            what_to_save['cost_normalizer'] = cost_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key(
            'Metrics/EpRet',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        # for u in range(self._cfgs.env_cfgs.U):
        #     self._logger.register_key(
        #         f'Metrics/EpCost{u}',
        #         window_length=self._cfgs.logger_cfgs.window_lens,
        #     )
        #     self._logger.register_key(f'Metrics/surr_cost_Adv{u}')
        self._logger.register_key(
            f'Metrics/EpCost',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )
        self._logger.register_key(f'Metrics/surr_cost_Adv')

        self._logger.register_key(
            'Metrics/EpLen',
            window_length=self._cfgs.logger_cfgs.window_lens,
        )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')
        self._logger.register_key('Loss/Loss_total', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')
        self._logger.register_key('Time/Steps')

        # register environment specific keys
        # for env_spec_key in self._env.env_spec_keys:
        #     self.logger.register_key(env_spec_key)

    def learn(self) -> tuple[list, list, list, str]:

        start_time = time.time()
        self._logger.log('INFO: Start training')

        relpath = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        if self._cfgs.seed is not None:
            relpath = f'seed-{str(self._cfgs.seed).zfill(3)}-{relpath}'

        for epoch in range(self._cfgs.train_cfgs.epochs):

            print(f"now epoch {epoch + 1}")

            # self._logger.append({})

            epoch_time = time.time()

            rollout_time = time.time()
            # self._logger[epoch]['Value/cost'] = []
            # self._logger[epoch]['Value/reward'] = []
            self._rollout(
                steps_per_epoch=self._cfgs.algo_cfgs.steps_per_epoch,
                # agent=self._actor_critic,
                # environment=self._env,
                buffer=self._buf,
                epoch=epoch,
                logger=self._logger,
            )
            # self._logger[epoch]['Time/Rollout'] = time.time() - rollout_time
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            # self._logger[epoch]['Loss/Loss_reward_critic'] = []
            # self._logger[epoch]['Loss/Loss_cost_critic'] = []
            # self._logger[epoch]['Train/Entropy'] = []
            # self._logger[epoch]['Train/PolicyRatio'] = []
            # self._logger[epoch]['Train/PolicyStd'] = []
            # self._logger[epoch]['Loss/Loss_pi'] = []
            self._update(epoch)
            # self._logger[epoch]['Time/Update'] = time.time() - update_time
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            # if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
            #     save = {
            #         "actor_param": self._actor_critic.actor.state_dict(),
            #         "actor_opt": self._actor_critic.actor_optimizer.state_dict(),
            #         "r_critic_param": self._actor_critic.reward_critic.state_dict(),
            #         "r_critic_opt": self._actor_critic.reward_critic_optimizer.state_dict(),
            #         "c_critic_param": self._actor_critic.cost_critic.state_dict(),
            #         "c_critic_opt": self._actor_critic.cost_critic_optimizer.state_dict()
            #     }
            #     for key, value in self._env.save().items():
            #         save[key] = value
            #     for key, value in self._logger[epoch].items():
            #         save[key] = value
            #     path = os.path.join(self._cfgs.logger_cfgs.log_dir, relpath, 'torch_save', f'epoch-{epoch + 1}.pt')
            #     os.makedirs(os.path.dirname(path), exist_ok=True)
            #     torch.save(save, path)
            #     print(f'epoch {epoch + 1} model saved')
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        # ep_ret = self._metrics['EpRet']
        # ep_cost = self._metrics['EpCost']
        # ep_len = self._metrics['EpLen']

        # ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        # # ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        # ep_cost = []
        # for u in range(self._cfgs.env_cfgs.U):
        #     ep_cost.append(self._logger.get_stats(f'Metrics/EpCost{u}')[0])
        # ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        # 返回保存位置
        path = os.path.join(self._cfgs.logger_cfgs.log_dir, self._cfgs.exp_name, relpath)

        # 保存logger
        # filename = os.path.join(path, "logger.json")
        # with open(filename, 'w') as file:
        #     json.dump(self._logger, file)
        #
        # # 保存metrics
        # filename = os.path.join(path, "metrics.json")
        # metrics = {}
        # for key, value in self._metrics.items():
        #     metrics[key] = [ele.tolist() if ele.shape[0] != 1 else ele.item() for ele in value]
        # with open(filename, 'w') as file:
        #     json.dump(metrics, file)

        return self._ret, self._cost, self._len, path

    def _rollout(self, steps_per_epoch, buffer, epoch, logger):

        self._reset_log()

        obs, info = self._env.reset(epoch * steps_per_epoch)

        ue_id = 0
        if epoch * steps_per_epoch % (480 * 700) > 0:
            ue_id = 1

        steps_time = 0
        for step in track(
                range(steps_per_epoch),
        ):

            # 训练到后半程时限制探索
            # if epoch <= self._cfgs.train_cfgs.epochs/2:
            #     act, value_r, value_c, logp = self._actor_critic.step(obs)
            # else:
            #     act, value_r, value_c, logp = self._actor_critic.step(obs, deterministic=True)

            # 全程确定性动作
            # act, value_r, value_c, logp = self._actor_critic.step(obs, deterministic=True)

            # 全程采样
            # TODO: agent每次只服务单用户
            act, value_r, value_c, logp = self._actor_critic.step(obs[ue_id])

            step_time_start = time.time()

            next_obs, reward, cost, info = self._env.step(act)

            steps_time += time.time() - step_time_start

            # self._log_value(reward=info['origin_reward'], cost=info['origin_cost'])
            #
            # if self._cfgs.algo_cfgs.use_cost:
            #     self._logger[epoch]['Value/cost'].append(value_c.tolist())
            # self._logger[epoch]['Value/reward'].append(value_r.tolist())
            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs[ue_id],
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                # last_value_r = torch.zeros(1)
                # last_value_c = torch.zeros(1)
                _, last_value_r, last_value_c, _ = self._actor_critic.step(obs[ue_id])
                # last_value_r = last_value_r.unsqueeze(0)
                # last_value_c = last_value_c.unsqueeze(0)

                self._log_metrics(logger)
                self._reset_log()

                buffer.finish_path(last_value_r, last_value_c)

                self._logger.store({'Time/Steps': steps_time})

    def _log_metrics(self, logger: Logger) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)

        # for u in range(self._cfgs.env_cfgs.U):
        #     logger.store(
        #         {
        #             f'Metrics/EpCost{u}': self._ep_cost[u].detach().cpu(),
        #         },
        #     )
        logger.store(
            {
                f'Metrics/EpCost': self._ep_cost.detach().cpu(),
            },
        )
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret.item(),
                'Metrics/EpLen': self._ep_len.item(),
            },
        )
        self._ret.append(self._ep_ret.item())
        self._cost.append(self._ep_cost.detach().cpu().tolist())
        self._len.append(self._ep_len.item())

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        # self._ep_ret += info.get('original_reward', reward) * (self._cfgs.algo_cfgs.gamma ** self._ep_len)
        # self._ep_cost += info.get('original_cost', cost) * (self._cfgs.algo_cfgs.gamma ** self._ep_len)
        self._ep_ret += info.get('original_reward', reward)
        self._ep_cost += info.get('original_cost', cost)
        self._ep_len += 1

    def _reset_log(self) -> None:
        """Reset the episode return, episode cost and episode length.
        """
        self._ep_ret = torch.zeros(1).to(self._device)
        self._ep_cost = torch.zeros(self._env.get_cost_num()).to(self._device)
        self._ep_len = torch.zeros(1).to(self._device)

    #
    # def _log_value(
    #     self,
    #     reward: torch.Tensor,
    #     cost: torch.Tensor,
    # ) -> None:
    #
    #     self._ep_ret += reward
    #     self._ep_cost += cost
    #     self._ep_len += 1
    #     self._ep_max_cost = torch.maximum(cost, self._ep_max_cost)
    #
    # def _log_metrics(self) -> None:
    #
    #     self._metrics['EpRet'].append(self._ep_ret)
    #     self._metrics['EpCost'].append(self._ep_cost)
    #     self._metrics['EpLen'].append(self._ep_len)
    #     self._metrics['EpMaxCost'].append(self._ep_max_cost)
    #
    # def _reset_log(self) -> None:
    #
    #     self._ep_ret = torch.zeros(1).to(self._device)
    #     self._ep_cost = torch.zeros(self._env.get_cost_num()).to(self._device)
    #     self._ep_len = torch.zeros(1).to(self._device)
    #     self._ep_max_cost = torch.zeros(self._env.get_cost_num()).to(self._device)
    #     for idx in range(self._env.get_cost_num()):
    #         self._ep_max_cost[idx] = -10000

    def _update(self, epoch) -> None:

        # print(f'epoch ave cost in epoch {epoch + 1} is '
        #       f'{(self._metrics["EpCost"][epoch].cpu() / self._metrics["EpLen"][epoch].cpu()).tolist()}')

        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        # TODO: 简单修改代码，让critic训练次数为actor的两倍
        for i in track(range(self._cfgs.algo_cfgs.update_iters * 2), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                if i % 2 == 1:
                    self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )

            final_kl = kl.item()
            update_counts += 1

            # if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
            #     print(f'Early stopping at iter {i + 1} due to reaching max kl')
            #     break
            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        # self._logger[epoch]['Train/StopIter'] = update_counts
        # self._logger[epoch]['Value/Adv'] = adv_r.mean().item()
        # self._logger[epoch]['Train/KL'] = final_kl
        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:

        self._actor_critic.reward_critic_optimizer.zero_grad()
        # TODO: 直接传参给critic，效果是做预测吗？
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        # 正则化
        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        # distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        # self._logger[epoch]['Loss/Loss_reward_critic'].append(loss.mean().item())
        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:

        self._actor_critic.cost_critic_optimizer.zero_grad()
        # TODO：修改了cost维度，使用一个网络同时拟合各个用户的cost
        cost_predict = self._actor_critic.cost_critic(obs)
        # loss = nn.functional.mse_loss(torch.stack(cost_predic, dim=0).t(), target_value_c)
        cost_predict = torch.stack(cost_predict, dim=0).t()
        loss = 0.0
        for i in range(self._cost_num):
            # 独立计算每个Cost的MSE
            loss_i = nn.functional.mse_loss(cost_predict[:, i], target_value_c[:, i])
            loss += loss_i  # 默认等权求和

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        # distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        # self._logger[epoch]['Loss/Loss_cost_critic'].append(loss.mean().item())
        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        # epoch
    ) -> None:

        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        # distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _compute_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:

        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
        # epoch
    ) -> torch.Tensor:

        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        # self._logger[epoch]['Train/Entropy'].append(entropy)
        # self._logger[epoch]['Train/PolicyRatio'].append(ratio.tolist())
        # self._logger[epoch]['Train/PolicyStd'].append(std)
        # self._logger[epoch]['Loss/Loss_pi'].append(loss.mean().item())
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

        return loss

    def get_param(self):
        # 打印模型参数以检查初始化情况
        print("Model parameters after initialization:")
        for name, param in self._actor_critic.actor.named_parameters():
            print(f"{name}: {param}")

