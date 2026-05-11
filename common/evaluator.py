import torch
import os
import json
import matplotlib.pyplot as plt
from omnisafe.utils.tools import get_device
from omnisafe.utils.config import Config

from agents.ActorCritic.base import Actor
from agents.ActorCritic.gaussian_learning_actor import GaussianLearningActor
from agents.ActorCritic.binary_learning_actor import BinaryLearningActor
# from env import environment as env
from env import environment_multicell_QuaDRiGa_SU as env
from common.tools import plot, plot_hist, plot_bar


class Evaluator:

    def __init__(self, cfgs: Config, save_filepath: str) -> None:

        self._cfgs: Config = cfgs
        self._save = save_filepath

        try:
            self._model_params = torch.load(save_filepath)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        assert hasattr(cfgs.train_cfgs, 'device'), 'Please specify the device in the config file.'
        self._device: torch.device = get_device(self._cfgs.train_cfgs.device)

        self._init_env()
        self._init_model()

        self._logger = []
        self._metrics = {
            'EpRet': [],
            'EpCost': [],
            'EpLen': []
        }

    def _init_env(self):
        self._env: env.Environment = env.Environment(
            self._cfgs.env_cfgs,
            self._device
        )
        if self._cfgs.env_cfgs.obs_normalize:
            param = self._model_params.get('obs_normalizer', None)
            # od_dict = dict(param)
            if param:
                self._env.load_norm(obs_norm_param=param)
            else:
                print('can not find obs norm params')
        if self._cfgs.env_cfgs.reward_normalize:
            param = self._model_params.get('reward_normalizer', None)
            if param:
                self._env.load_norm(reward_norm_param=param)
            else:
                print('can not find reward norm params')
        if self._cfgs.env_cfgs.cost_normalize:
            param = self._model_params.get('cost_normalizer', None)
            if param:
                self._env.load_norm(cost_norm_param=param)
            else:
                print('can not find cost norm params')

    def _init_model(self):

        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._actor: Actor = GaussianLearningActor(
                obs_space_dim=self._env.get_obs_dim(),
                act_space_dim=self._env.get_act_dim(),
                hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
                activation=self._cfgs.model_cfgs.actor.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode
            ).to(self._device)
        if self._cfgs.model_cfgs.actor_type == 'binary_learning':
            self._actor: Actor = BinaryLearningActor(
                obs_space_dim=self._env.get_obs_dim(),
                act_space_dim=self._env.get_act_dim(),
                hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
                activation=self._cfgs.model_cfgs.actor.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode
            ).to(self._device)
        #
        # self._actor: Actor = GaussianLearningActor(
        #     obs_space_dim=self._env.get_obs_dim(),
        #     act_space_dim=self._env.get_act_dim(),
        #     hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
        #     activation=self._cfgs.model_cfgs.actor.activation,
        #     weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode
        # ).to(self._device)
        self._actor.load_state_dict(self._model_params['actor_param'])

    def process_sinr_data(self, sinr_data, period):
        result = []
        # t = sinr_data[0::period]
        for i in range(period):
            temp = []
            ave_data = sinr_data[i::period]
            for j in range(self._cfgs.env_cfgs.U):
                ave = [data[j] for data in ave_data]
                ave = sum(ave) / len(ave)
                temp.append(ave)
            result.append(temp)
        return result

    def evaluate(self, step_num, need_plot: bool = True):

        step_rewards: list[float] = []
        step_costs: list[float] = []
        slot_bits = []
        user_bits = []
        user_BLER = []
        user_BLER_ideal = []
        user_OLLA = []
        user_sinr = []
        user_mcs = []
        user_mcs_ideal = []
        postsinr_estimation = []
        postsinr_estimation_raw = []
        user_layer = []
        pic_save_path = os.path.dirname(os.path.dirname(self._save))
        ep_cost = torch.zeros(self._env.get_cost_num()).to(self._device)
        total_bits = 0

        obs, info = self._env.reset()

        for step in range(step_num):

            # with torch.no_grad():
            #     if self._actor is not None:
            #         act = self._actor.predict(
            #             obs,
            #             deterministic=True,
            #         )
            #     else:
            #         raise ValueError(
            #             'The policy must be provided or created before evaluating the agent.',
            #         )
            act = []
            with torch.no_grad():
                for ue_id in range(self._cfgs.env_cfgs.U):
                    if self._actor is not None:
                        act.append(
                            self._actor.predict(
                                obs[ue_id],
                                deterministic=True,
                            )
                        )
                    else:
                        raise ValueError(
                            'The policy must be provided or created before evaluating the agent.',
                        )
            act = torch.cat(act)

            obs, reward, cost, info = self._env.step(act)

            ep_cost += info.get('original_cost', cost)
            print(f'step {step} complete')

            step_rewards.append(info['origin_reward'].item())
            step_costs.append(info['cost_list'])
            slot_bits.append(info['tot_bits'])
            user_bits.append(info['user_bits'])
            user_BLER.append(info['user_BLER'])
            user_BLER_ideal.append(info['user_BLER_ideal'])
            user_OLLA.append(info['user_OLLA'])
            user_sinr.append(info['user_sinr'])
            user_mcs.append(info['user_mcs'])
            user_mcs_ideal.append(info['user_mcs_ideal'])
            postsinr_estimation.append(info['postsinr_estimation'])
            postsinr_estimation_raw.append(info['postsinr_estimation_raw'])
            total_bits += info['tot_bits']
            user_layer.append(info['user_layer'])

        # sinr_condition = info['sinr_condition']
        # plot_hist(sinr_condition[0], 'MRT user 0 gain condition', 'gain/dB')
        # plot_hist(sinr_condition[1], 'user 1 SINR condition')
        # plot_hist(sinr_condition[2], 'user 2 SINR condition')
        # print(ep_cost)
        user_MCS_distribution = info['user_MCS_distribution']
        user_sinr_ave = self.process_sinr_data(user_sinr, 160)
        postsinr_ave = self.process_sinr_data(postsinr_estimation, 160)

        slots = range(step_num)
        data_dict = {
            "step_rewards": step_rewards,
            "step_costs": step_costs,
            "slots": list(slots),
            "slot_bits": slot_bits,
            "user_bits": user_bits,
            "user_BLER": user_BLER,
            "user_BLER_ideal": user_BLER_ideal,
            "user_OLLA": user_OLLA,
            "user_sinr": user_sinr,
            "user_mcs": user_mcs,
            "user_mcs_ideal": user_mcs_ideal,
            "postsinr_estimation": postsinr_estimation,
            "postsinr_estimation_raw": postsinr_estimation_raw,
            "user_layer": user_layer,
        }

        data_file_name = "eval_data.json"
        data_file_path = os.path.join(pic_save_path, data_file_name)
        # 保存为JSON
        with open(data_file_path, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)  # indent=4 格式化输出，增强可读性

        if need_plot:
            plot(step_costs, slots, 'costs per slot', pic_save_path)
            plot(slot_bits, slots, 'total bits per slot', pic_save_path)
            plot(user_bits, slots, 'bits per UE per slot', pic_save_path)
            plot(user_BLER, slots, 'BLER per UE per slot', pic_save_path)
            plot(user_BLER_ideal, slots, 'ideal mean BLER per UE per slot', pic_save_path)
            plot(user_OLLA, slots, 'OLLAs per UE per slot', pic_save_path)
            plot(user_sinr, slots, 'sinr per UE per slot', pic_save_path)
            plot(postsinr_estimation, slots, 'postsinr estimation per UE per slot', pic_save_path)
            plot(user_sinr_ave, range(160), 'ave sinr per UE per slot', pic_save_path)
            plot(postsinr_ave, range(160), 'ave postsinr estimation per UE per slot', pic_save_path)
            for i in range(len(user_MCS_distribution)):
                plot_bar([x / step_num for x in user_MCS_distribution[i]], None,
                         f'user {i} MCS distribution', "MCS order", pic_save_path)
            # print(f'after slots {step_num}, final SE is {final_SE}')
            print(
                f'after slots {step_num}, average bits/s/Hz is {total_bits / step_num}, BLER is about {user_BLER[-1]}')

            user_num = len(data_dict['user_sinr'][0])
            user_sinr2 = [[] for _ in range(user_num)]
            # user_mcs = [[] for _ in range(user_num)]
            # user_mcs_ideal = [[] for _ in range(user_num)]
            postsinr_estimation2 = [[] for _ in range(user_num)]
            postsinr_estimation_raw2 = [[] for _ in range(user_num)]
            for u in range(user_num):
                user_sinr2[u] = [sinr[u] for sinr in data_dict['user_sinr']]
                # user_mcs[u] = [mcs[u] for mcs in data_dict['user_mcs']]
                # user_mcs_ideal[u] = [mcs[u] for mcs in data_dict['user_mcs_ideal']]
                postsinr_estimation2[u] = [postsinr[u] for postsinr in data_dict['postsinr_estimation']]
                postsinr_estimation_raw2[u] = [postsinr_raw[u] for postsinr_raw in data_dict['postsinr_estimation_raw']]

                plt.figure()
                plt.plot(user_sinr2[u], label="real sinr")
                plt.plot(postsinr_estimation2[u], label="estimated sinr")
                plt.plot(postsinr_estimation_raw2[u], label="estimated sinr (without OLLA)")
                # plt.plot(user_mcs[u], label="mcs")
                plt.grid(True)
                plt.legend()
                plt.title(f"sinr condition of user {u}")

                filename = f"sinr condition of user {u}.png"
                filepath = os.path.join(pic_save_path, filename)
                plt.savefig(filepath)

                plt.show()

        return data_dict

    def plot_pos(self):
        self._env.plot_points()
