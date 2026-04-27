# import torch
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import h5py

from common.tools import get_dis, determine_rows_cols

from omnisafe.utils.config import Config
# from omnisafe.common.normalizer import Normalizer

import os
import json
from omnisafe.utils.tools import get_device
from omnisafe.utils.config import Config
from common.tools import plot, plot_bar
from common.tools import get_default_kwargs_yaml
from common.tools import BiasCorrectedEWMA

from env.environment_multicell_QuaDRiGa_SU import QuadrigaMultiCellChannelSource, User, BS, FeedbackScheduler

class myBS(BS):
    def choose_mcs2(self, mcs_table: dict, ceiling: bool, mean_SINR_estimate: bool, BSs):
        mcs_list = []
        postSINR_estimation_list = []
        postSINR_estimation_raw_list = []
        gain_list = []
        interference_list = []
        for u in self.serve_UEs:
            gain_list.append([])
            interference_list.append([])
            if not ceiling:
                if mean_SINR_estimate:
                    sinr_estimate_list = []
                    for l in range(u.n_layer):
                        gain = ((self.rho[u.id] ** (2 * self.CSI_update_delay)) * (self.H_l_bs_serve[u.id] ** 2)
                                * self.P_user[u.id][l] * u.combiner_eff_gain[l] * self.Mt)
                        interference = 0
                        for v in self.serve_UEs:
                            for i in range(v.n_layer):
                                if v == u and i == l:
                                    continue
                                elif v == u:
                                    interference += (self.P_user[v.id][i] * u.combiner_eff_gain[l])
                                else:
                                    interference += (self.P_user[v.id][i] * u.combiner_eff_gain[l]
                                                 * np.trace(self.Rt[v.id] @ self.Rt[u.id]).real / self.Mt)
                        mu_loss = ((1 - self.rho[u.id] ** (2 * self.CSI_update_delay)) * (self.H_l_bs_serve[u.id] ** 2)
                                   * interference)
                        for bs_id, hl in u.large_scale_fadings.items():
                            if bs_id != self.id:
                                mu_loss += (u.large_scale_fadings[bs_id] ** 2) * self.P
                        sinr_estimate_list.append(gain / (mu_loss + self.noise))
                        gain_list[u.id].append(10 * np.log10(gain))
                        interference_list[u.id].append(10 * np.log10(mu_loss))
                    sinr_estimate = np.exp(np.mean(np.log(np.array(sinr_estimate_list))))       # 层间几何平均
                    sinr_estimate = 10* np.log10(sinr_estimate)
                else:
                    gain = (np.linalg.norm(self.H_bs_serve[u.id][0], 'fro') ** 2
                            / (self.H_bs_serve[u.id][0].shape[0] * self.H_bs_serve[u.id][0].shape[1]))
                    sinr_estimate = 10 * np.log10(gain * self.P / self.noise)
                    mu_loss = self.mu_loss_per_user_dB * (len(self.serve_UEs) - 1)
                    layer_loss = 10 * np.log10(u.n_layer)
                    sinr_estimate = sinr_estimate - mu_loss - layer_loss
                    gain_list[u.id].append(10 * np.log10(gain))
                    interference_list[u.id].append(mu_loss)

                postSINR_estimation_raw_list.append(sinr_estimate)
                sinr_estimate += self.OLLA[u.id]
            else:
                sinr_list = []
                for l in range(u.n_layer):
                    up = 0
                    down = 0
                    combiner = u.combiner[l, :]

                    for bs in BSs:
                        for v in bs.serve_UEs:
                            for i in range(v.n_layer):
                                precoding_vector = v.precoder[:, i]

                                if v == u and i == l:
                                    up = np.linalg.norm(combiner @ bs.H_bs_real[u.id][0].conj().T @ precoding_vector) ** 2
                                else:
                                    down += np.linalg.norm(combiner @ bs.H_bs_real[u.id][0].conj().T @ precoding_vector) ** 2

                    sinr_list.append(up / (down + self.noise))
                sinr_estimate = np.exp(np.mean(np.log(np.array(sinr_list))))  # 层间几何平均
                sinr_estimate = 10 * np.log10(sinr_estimate)
                postSINR_estimation_raw_list.append(sinr_estimate)
                # print(f"sinr_estimate is {sinr_estimate}")
            postSINR_estimation_list.append(sinr_estimate)
            # TODO: 调整MCS映射表也动了这里的初始化
            mcs = "1"
            for key, value in mcs_table.items():
                if sinr_estimate >= value[1]:
                    mcs = key
                else:
                    break
            mcs_list.append(mcs)
        info = {
            "postSINR_estimation_list": postSINR_estimation_list,
            "postSINR_estimation_raw_list": postSINR_estimation_raw_list,
            "gain_list": gain_list,
            "interference_list": interference_list,
        }
        return mcs_list, info

class Environment:

    def __init__(self, cfg: Config, device):

        self._cfg = cfg
        self._device = device

        # 环境基本设置
        self._B = cfg.B              # 基站数量
        self._K = cfg.U              # 用户数量(单BS内)
        self._Mt = cfg.Mt              # BS天线数量
        self._Mr = cfg.Mr               # UE天线数量
        self._Bw = cfg.Bw            # 单个子载波带宽(kHz)
        self._N = cfg.N              # 每个时隙的OFDM符号数
        self._K_all = self._K * self._B # 总UE数量

        # 信道模型参数
        self._fc = cfg.fc               # 中心频率(MHz)
        self._d0 = cfg.d0               # 大尺度衰落模型参数
        self._d1 = cfg.d1
        self._d2 = cfg.d2
        self._r = cfg.r                 # 信道空间相关系数
        self._rho = cfg.rho             # 信道时间相关系数
        self._rho2 = np.sqrt(1 - self._rho ** 2)
        self._delta_theta = cfg.delta_theta / 180 * np.pi  # 围绕中心角向两边扩散的最大角度
        self._kai = cfg.kai  # 2*pi*d/λ，固定取d/λ为1/2
        # self._sigma_sh = cfg.sigma_sh   # shadow fading(dB)
        # self._delta_sh = cfg.delta_sh   # shadow fading参数

        self._noise = 10 ** (cfg.noise/10 - 3)              # 噪声功率(W/Hz)
        self._sigma = self._noise * self._Bw * 1e3 * 12          # 噪声功率(W)  TODO: 还乘了每个RB的子载波数

        self._OFDM_t = 1 / (self._Bw * 1e3)                 # OFDM符号时长
        self._slot_t = cfg.t_slot                           # 时隙时长(ms)
        # self._f_carrier = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 for n in range(self._F)]   # 各载波频率(MHz)
        # self._f_subband = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 * self._C for n in range(self._K_all)]   # 各子带中心频率(MHz)
        self._H = {}                                      # 信道矩阵
        self._P = 10 ** (cfg.P/10 - 3)                                   # 固定发射功率分配(W)

        # # 测试有干扰情况下的sinr分布情况
        # # 用户数*用户数，第一个参数表明是哪个用户，第二个参数表明存在多少用户干扰
        # self.sinr_interference_condition = [[[] for _ in range(self._K_all)] for _ in range(self._K_all)]

        # 用户设置参数
        self._h_U = cfg.h_U                            # 用户高度
        self._BLER_T = cfg.BLER_T                      # 用户BLER阈值
        self._N_layer = cfg.N_layer              # 用户数据流数
        # shadow_user = np.random.normal(0, 1, size=self._K_all)     # User侧的shadow fading参数

        # BS设置参数
        # self._max_P = cfg.max_P            # 每个BS所能承受的最大功率单元数
        # self._dP = 10 ** (cfg.max_P_dBm / 10 - 3) / cfg.max_P           # 功率分配粒度(W)
        self._h_BS = cfg.h_BS              # BS高度
        self._K_BS = cfg.K_BS              # BS最大连接用户数
        self._SRS_period = cfg.SRS_period  # SRS更新周期
        self._buffer_len = cfg.buffer_len  # BS存储历史数据条数
        self._D = cfg.feedback_delay       # ACK/NACK反馈延迟
        self._BS_pos = cfg.BS_pos          # BS位置
        # shadow_BS = np.random.normal(0, 1, size=self._B)       # BS侧的shadow fading参数

        # self._Users = cfg["Users"]      # 用户数据
        # self._BSs = cfg["BSs"]          # BS数据

        self._region_bound = cfg.region_bound                # 所考虑区域大小
        self._min_distance_BS = cfg.min_distance_BS          # BS间最小间距
        self._min_distance_User = cfg.min_distance_User      # 用户间最小间距

        self.MCS_table = {
            # MCS阶数: [编码效率(bits/symbol), SINR阈值(dB)]
            # SINR大于阈值则可以选择对应MCS，此时最优MCS为SINR恰大于本阶阈值而小于下阶阈值
            '0': [0.2344, -6.05],
            '1': [0.3770, -4.07],
            '2': [0.6016, -1.93],
            '3': [0.8770, 0.01],
            '4': [1.1758, 1.69],
            '5': [1.4766, 3.13],
            '6': [1.6953, 4.09],
            '7': [1.9141, 4.99],
            '8': [2.1602, 5.94],
            '9': [2.4063, 6.86],
            '10': [2.5703, 7.45],
            '11': [2.7305, 8.01],
            '12': [3.0293, 9.04],
            '13': [3.3223, 10.02],
            '14': [3.6094, 10.96],
            '15': [3.9023, 11.91],
            '16': [4.2129, 12.90],
            '17': [4.5234, 13.88],
            '18': [4.8164, 14.79],
            '19': [5.1152, 15.72],
            '20': [5.3320, 16.39],
            '21': [5.5547, 17.07],
            '22': [5.8906, 18.10],
            '23': [6.2266, 19.13],
            '24': [6.5703, 20.17],
            '25': [6.9141, 21.22],
            '26': [7.1602, 21.96],
            '27': [7.4063, 22.21],
        }

        self.feedback_scheduler = FeedbackScheduler(self._D)

        # 是否使用SINR均值老化动态估计
        self._mean_SINR_estimate = cfg.mean_SINR_estimate
        # 是否L2获取L1实际预编码
        self._ceiling = cfg.ceiling
        # 是否传统OLLA机制
        self._OLLA_scheme = cfg.OLLA
        # 是否读取固定信道实现
        self.fix_channel = cfg.fix_channel

        if self.fix_channel:
            # quadriga信道文件地址
            self._quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/QuaDriGa/quadriga_multicell_channel_out"
        else:
            self._quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/QuaDriGa/quadriga_UE_packs_test"
        self._qg_src = QuadrigaMultiCellChannelSource(self._quadriga_dir)

        self.feedback_scheduler = FeedbackScheduler(self._D)

        # 生成用户
        self.UEs: list[User] = []
        for i in range(self._K_all):
            user = User(i, self._BLER_T, self._Mr, self._N_layer)
            self.UEs.append(user)
            self.feedback_scheduler.log_user(user)

        # 生成BS
        self.BSs: list[myBS] = []
        for i in range(self._B):
            # temp = {"max P": self._max_P, "location": BS_pos[i], "shadow": shadow_BS[i]}
            self.BSs.append(myBS(i, self._P, self._sigma, self._K_BS, self._Mt, self._Mr,
                               self._SRS_period, self._buffer_len, self.MCS_table))

        # 记录用户MCS选择分布情况
        self.user_MCS_distribution = []
        for i in range(self._K):
            self.user_MCS_distribution.append([0] * len(self.MCS_table.keys()))

        self._slots = 0

        self._tot_bits = 0
        self.state = None

    def reset(self):

        self._slots = 0
        self._tot_bits = 0

        # 读取quadriga信道
        self._H = {}
        for bs in self.BSs:
            self._H[bs.id] = {}
            for u in self.UEs:
                self._H[bs.id][u.id] = self._qg_src.get_H(bs.id, u.id, self._slots)  # [NRB, NBS, N_RX]
                assert (self._H[bs.id][u.id].shape[1], self._H[bs.id][u.id].shape[2]) == (self._Mt, self._Mr), "channel shape mismatch!"

        for bs in self.BSs:
            bs.reset()

        # 设置BS与AP关联
        for u in self.UEs:
            bs_id = u.id // self._K  # 先固定UE和BS配对，每个BS配K个UE
            self.BSs[bs_id].log_user(u)
            u.serve_BS = self.BSs[bs_id]
            self.feedback_scheduler.reset(u)

        # 各个BS整理信道信息
        # 只有SRS周期时BS才会更新信道
        for bs in self.BSs:
            bs.collect_channels(self._H[bs.id], self._slots, self.UEs)

        info = {}

        state = None

        return state, info

    def step(self):
        """
        环境交互函数，开始时为当前slot进行发送的阶段，根据action发送完成记录数据后开启下一个slot的状态转换，再将状态交给agent进行决策

        :returns:
            state: 输入到agent的状态，包括CSI、历史ACK/NACK、更新延迟
            reward: 单时隙奖励
            cost: 单时隙成本，由每个UE的成本拼接而成
            info: 其他可能需要的信息
        """

        self._slots += 1

        # rank自适应并计算预编码
        for bs in self.BSs:
            bs.optimize_n_layer_exhaustive(self.MCS_table, self._mean_SINR_estimate)
            # bs.generate_precoder()

        layer_sets = [u.n_layer for u in self.UEs]

        # TODO: 尽管是多小区，但只需要对0号BS进行真实传输，其余BS只是用来生成干扰的
        MCS_list, info = self.BSs[0].choose_mcs2(self.MCS_table, self._ceiling, self._mean_SINR_estimate, self.BSs)
        postSINR_estimation_list = info["postSINR_estimation_list"]
        postSINR_estimation_raw_list = info["postSINR_estimation_raw_list"]
        gain_list = info["gain_list"]
        interference_list = info["interference_list"]

        # print(f"MCS_list: {MCS_list}")

        # 根据调度结果更新用户数据队列，记录这一时隙内总发送比特数，并考察时延约束违反情况
        tot_bits = 0
        success_users = self._K
        ACK_list = []
        user_bits = []
        user_BLER = []
        user_BLER_ideal = []
        user_OLLA = []
        user_sinr = []
        user_gain = []
        user_interference = []
        user_interference_plus_noise = []
        for i, u in enumerate(self.BSs[0].serve_UEs):

            bits, ACK, info = self.get_rate(u, str(MCS_list[i]))
            ideal_bler = self.get_bler(10 ** (info["sinr"] / 10), self.MCS_table[MCS_list[i]][0])
            delayed_feedback = self.feedback_scheduler.update(u, ACK)
            u.serve_BS.update_ACK(u, delayed_feedback, self._OLLA_scheme)
            u.update_BLER(ACK, self._slots, ideal_bler)
            # u.update_BLER(ACK, self._slots)

            # 记录各类参数
            tot_bits += bits / self._K
            user_bits.append(bits)
            ACK_list.append(ACK)
            user_BLER.append(u.BLER)
            user_BLER_ideal.append(u.BLER_ideal)
            user_OLLA.append(u.serve_BS.OLLA[u.id])
            user_sinr.append(info["sinr"])
            user_gain.append(info["gain"])
            user_interference.append(info["interference"])
            user_interference_plus_noise.append(info["noise + interference"])
            self.user_MCS_distribution[i][int(MCS_list[i]) - 1] += 1

        # tot_bps = tot_bits / (self._slot_t * 1e-3)
        # user_bps = [ele / (self._slot_t * 1e-3) for ele in user_bits]

        # 完美MCS选择
        ideal_MCS_list = []
        for i, u in enumerate(self.BSs[0].serve_UEs):
            mcs = "1"
            for key, value in self.MCS_table.items():
                if user_sinr[i] >= value[1]:
                    mcs = key
                else:
                    break
            ideal_MCS_list.append(mcs)

        # 奖励
        # 直接以总传输bit数为奖励
        reward = tot_bits / 10

        # # 成本
        # cost_list = []
        # # 直接用瞬时ACK/NACK情况结合BLER目标计算cost
        # for i, u in enumerate(self.UEs):
        #     cost_u = (1 - ACK_list[i]) - u.BLER_T
        #     cost_list.append(cost_u)

        # 本时隙结束，下一时隙开始
        # 读取quadriga信道
        self._H = {}
        for bs in self.BSs:
            self._H[bs.id] = {}
            for u in self.UEs:
                self._H[bs.id][u.id] = self._qg_src.get_H(bs.id, u.id, self._slots)  # [NRB, NBS, N_RX]
                assert (self._H[bs.id][u.id].shape[1], self._H[bs.id][u.id].shape[2]) == (self._Mt, self._Mr), "channel shape mismatch!"

        # 各个BS整理信道信息
        # 只有SRS周期时BS才会更新信道
        for bs in self.BSs:
            bs.collect_channels(self._H[bs.id], self._slots, self.UEs)

        info = {
            'tot_bits': tot_bits,
            # 'cost_list': cost_list,
            'user_bits': user_bits,
            'user_BLER': user_BLER,
            'user_BLER_ideal': user_BLER_ideal,
            'user_OLLA': user_OLLA,
            'user_sinr': user_sinr,
            'user_mcs': [int(mcs) for mcs in MCS_list],
            'user_mcs_ideal': [int(mcs) for mcs in ideal_MCS_list],
            'user_gain': user_gain,
            'user_interference': user_interference,
            'user_interference_plus_noise': user_interference_plus_noise,
            'postsinr_estimation': postSINR_estimation_list,
            'postsinr_estimation_raw': postSINR_estimation_raw_list,
            'gain_estimation': gain_list,
            'interference_estimation': interference_list,
            'user_MCS_distribution': self.user_MCS_distribution,
        }

        state, cost = None, None

        # 返回内容：输入给agent的状态、奖励、成本、性能指标
        return state, reward, cost, info

    def generate_gaussian_channel(self, Rr, Rt, n_layers):

        Mt = self._Mt
        Mr = self._Mr

        # 步骤1：特征值分解（对 Hermitian 矩阵用 eigh，确保数值稳定）
        eigenvalues_t, eigenvectors_t = np.linalg.eigh(Rt)

        # 步骤2：处理特征值（确保非负，避免数值误差导致的微小负数）
        eigenvalues_t = eigenvalues_t[::-1]
        eigenvectors_t = eigenvectors_t[:, ::-1]
        epsilon = 1e-18
        eigenvalues_t = np.maximum(eigenvalues_t, epsilon)  # 替换负特征值为极小正数
        lambda_sqrt = np.sqrt(eigenvalues_t)  # 特征值开平方

        # 步骤3：构造平方根矩阵 U * Lambda^(1/2)
        Rt_square = eigenvectors_t @ np.diag(lambda_sqrt)

        eigenvalues_r, eigenvectors_r = np.linalg.eigh(Rr)

        eigenvalues_r = eigenvalues_r[::-1]
        eigenvectors_r = eigenvectors_r[:, ::-1]
        epsilon = 1e-18
        eigenvalues_r = np.maximum(eigenvalues_r, epsilon)
        lambda_sqrt = np.sqrt(eigenvalues_r)

        Rr_square = eigenvectors_r @ np.diag(lambda_sqrt)

        combiner_statistical = eigenvectors_r[:, :n_layers].conj().T
        combiner_equal_gain = lambda_sqrt[:n_layers]

        # 一次性生成所有随机数
        h_real = np.random.randn(1, Mt, Mr)
        h_imag = np.random.randn(1, Mt, Mr)
        h0 = (h_real + 1j * h_imag) / np.sqrt(2)

        # 批量变换
        h = Rt_square @ h0 @ Rr_square

        return h, combiner_statistical, combiner_equal_gain

    def get_bler(self, sinr, R):
        n = 12 * 14
        C = np.log2(1 + sinr)
        sqrt_V = np.log2(np.e) * np.sqrt((1 - 1 / ((1 + sinr) ** 2)) / n)
        return 0.5 * (1 - sp.erf(((C - R) / sqrt_V) / np.sqrt(2)))

    # 计算用户SINR
    def get_SINR(self, u: User):
        sinr_list = []
        up_list = []
        down_list = []
        for l in range(u.n_layer):
            up = 0
            down = 0
            combiner = u.combiner[l, :]

            for bs in self.BSs:
                for v in bs.serve_UEs:
                    for i in range(v.n_layer):
                        precoding_vector = v.precoder[:, i]

                        if v == u and i == l:
                            up = np.linalg.norm(combiner @ self._H[bs.id][u.id][0].conj().T @ precoding_vector) ** 2
                        else:
                            down += np.linalg.norm(combiner @ self._H[bs.id][u.id][0].conj().T @ precoding_vector) ** 2

            sinr_list.append(up / (down + self._sigma))
            up_list.append(up)
            down_list.append(down)
        sinr = np.exp(np.mean(np.log(np.array(sinr_list))))  # 层间几何平均

        info = {
            "gain": up_list,
            "interference": down_list,
            "noise + interference": [d + self._sigma for d in down_list],
        }

        return sinr, info

    # 计算某用户在时隙内的可发送比特数
    def get_rate(self, u: User, mcs: str):

        info = {}
        sinr, sinr_info = self.get_SINR(u)
        # if sinr != 0 and has_ICI:
        # if sinr != 0:
        # if sinr != 0:
        #     temp = 10 * np.log10(sinr)
        #     print(f"the SINR of user {u.id} is : {temp}")
            # self.sinr_interference_condition[u][has_ICI].append(temp)

        # ideal_bler = self.get_bler(sinr, self.MCS_table[mcs][0])
        # if np.random.rand() >= ideal_bler:
        #     ACK = 1
        #     bits = self.MCS_table[mcs][0] * self._N
        # else:
        #     ACK = 0
        #     bits = 0

        sinr = 10 * np.log10(sinr)
        # print(f"the SINR of user {u.id} is : {sinr}")
        if sinr >= self.MCS_table[mcs][1]:
            ACK = 1
            bits = self.MCS_table[mcs][0] * u.n_layer
        else:
            ACK = 0
            bits = 0

        info["sinr"] = sinr
        info["gain"] = [10 * np.log10(g) for g in sinr_info["gain"]]
        info["interference"] = [10 * np.log10(i) for i in sinr_info["interference"]]
        info["noise + interference"] = [10 * np.log10(ni) for ni in sinr_info["noise + interference"]]
        # print(f"ACK/NACK: {ACK}")

        return bits, ACK, info


class Evaluator:

    def __init__(self, cfgs: Config, save_filepath: str) -> None:

        self._cfgs: Config = cfgs
        self._save = save_filepath

        assert hasattr(cfgs.train_cfgs, 'device'), 'Please specify the device in the config file.'
        # self._device: torch.device = get_device(self._cfgs.train_cfgs.device)
        self._device = cfgs.train_cfgs.device

        self._init_env()

        self._logger = []
        self._metrics = {
            'EpRet': [],
            'EpCost': [],
            'EpLen': []
        }

    def _init_env(self):
        self._env = Environment(
            self._cfgs.env_cfgs,
            self._device
        )

    def process_sinr_data(self, sinr_data, period):
        result = []
        # t = sinr_data[0::period]
        for i in range(period):
            temp = []
            ave_data = sinr_data[i::period]
            for j in range(self._cfgs.env_cfgs.U):
                ave = [data[j] for data in ave_data]
                ave = np.array(ave)
                if len(ave.shape) == 1:
                    ave = ave.mean()
                    ave = float(ave)
                else:
                    ave = ave.mean(axis=0)
                    ave = [float(i) for i in ave.tolist()]
                temp.append(ave)
            result.append(temp)
        return result

    def evaluate(self, step_num, need_plot: bool = True):

        step_rewards: list[float] = []
        # step_costs: list[float] = []
        slot_bits = []
        user_bits = []
        user_BLER = []
        user_BLER_ideal = []
        user_OLLA = []
        user_sinr = []
        user_mcs = []
        user_mcs_ideal = []
        user_gain = []
        user_interference = []
        user_interference_plus_noise = []
        postsinr_estimation = []
        postsinr_estimation_raw = []
        gain_estimation = []
        interference_estimation = []
        pic_save_path = self._save
        # ep_cost = torch.zeros(self._env.get_cost_num()).to(self._device)
        total_bits = 0

        obs, info = self._env.reset()

        for step in range(step_num):

            obs, reward, cost, info = self._env.step()

            # ep_cost += info.get('original_cost', cost)
            print(f'step {step} complete')

            # step_rewards.append(info['origin_reward'].cpu())
            # step_costs.append(info['cost_list'])
            slot_bits.append(info['tot_bits'])
            user_bits.append(info['user_bits'])
            user_BLER.append(info['user_BLER'])
            user_BLER_ideal.append(info['user_BLER_ideal'])
            user_OLLA.append(info['user_OLLA'])
            user_sinr.append(info['user_sinr'])
            user_mcs.append(info['user_mcs'])
            user_mcs_ideal.append(info['user_mcs_ideal'])
            user_gain.append(info['user_gain'])
            user_interference.append(info['user_interference'])
            user_interference_plus_noise.append(info['user_interference_plus_noise'])
            postsinr_estimation.append(info['postsinr_estimation'])
            postsinr_estimation_raw.append(info['postsinr_estimation_raw'])
            gain_estimation.append(info['gain_estimation'])
            interference_estimation.append(info['interference_estimation'])

            total_bits += info['tot_bits']

        # sinr_condition = info['sinr_condition']
        # plot_hist(sinr_condition[0], 'MRT user 0 gain condition', 'gain/dB')
        # plot_hist(sinr_condition[1], 'user 1 SINR condition')
        # plot_hist(sinr_condition[2], 'user 2 SINR condition')
        # print(ep_cost)
        user_MCS_distribution = info['user_MCS_distribution']
        user_sinr_ave = self.process_sinr_data(user_sinr, 160)
        # user_gain_ave = self.process_sinr_data(user _gain, 160)
        # user_interference_ave = self.process_sinr_data(user_interference, 160)
        # user_interference_plus_noise_ave = self.process_sinr_data(user_interference_plus_noise, 160)
        # gain_estimation_ave = self.process_sinr_data(gain_estimation, 160)
        # interference_estimation_ave = self.process_sinr_data(interference_estimation, 160)
        # postsinr_ave = self.process_sinr_data(postsinr_estimation, 160)

        slots = range(step_num)
        data_dict = {
            "step_rewards": step_rewards,
            # "step_costs": step_costs,
            "slots": list(slots),
            "slot_bits": slot_bits,
            "user_bits": user_bits,
            "user_BLER": user_BLER,
            "user_BLER_ideal": user_BLER_ideal,
            "user_OLLA": user_OLLA,
            "user_MCS_distribution": user_MCS_distribution,
            "user_mcs": user_mcs,
            "user_mcs_ideal": user_mcs_ideal,
            "user_sinr": user_sinr,
            "user_sinr_ave": user_sinr_ave,
            "user_gain": user_gain,
            # "user_gain_ave": user_gain_ave,
            "user_interference": user_interference,
            # "user_interference_ave": user_interference_ave,
            "user_interference_plus_noise": user_interference_plus_noise,
            # "user_interference_plus_noise_ave": user_interference_plus_noise_ave,
            "postsinr_estimation": postsinr_estimation,
            "postsinr_estimation_raw": postsinr_estimation_raw,
            "gain_estimation": gain_estimation,
            "interference_estimation": interference_estimation,
            # "gain_estimation_ave": gain_estimation_ave,
            # "interference_estimation_ave": interference_estimation_ave,
            # "postsinr_ave": postsinr_ave,
        }

        if need_plot:
            # plot(step_costs, slots, 'costs per slot', pic_save_path)
            # plot(slot_bits, slots, 'total bits per slot', pic_save_path)
            # plot(user_bits, slots, 'bits per UE per slot', pic_save_path)
            plot(user_BLER, slots, 'BLER per UE per slot', pic_save_path)
            # plot(user_BLER_ideal, slots, 'ideal BLER per UE per slot', pic_save_path)
            plot(user_OLLA, slots, 'OLLAs per UE per slot', pic_save_path)
            # plot(user_sinr, slots, 'sinr per UE per slot', pic_save_path)
            # plot(postsinr_estimation, slots, 'postsinr estimation per UE per slot', pic_save_path)
            # plot(user_sinr_ave, range(160), 'ave sinr per UE per slot', pic_save_path)
            # plot(user_gain_ave, range(160), 'ave gain per UE per slot', pic_save_path)
            # plot(user_interference_ave, range(160), 'ave interference per UE per slot', pic_save_path)
            # plot(user_interference_plus_noise_ave, range(160), 'ave interference + noise per UE per slot', pic_save_path)
            # plot(postsinr_ave, range(160), 'ave postsinr estimation per UE per slot', pic_save_path)
            for i in range(len(user_MCS_distribution)):
                plot_bar([x / step_num for x in user_MCS_distribution[i]], None,
                         f'user {i} MCS distribution', "MCS order", pic_save_path)
            print(f'after slots {step_num}, average bits/s/Hz is {total_bits / step_num}, BLER is about {user_BLER[-1]}')

            user_num = len(data_dict['user_sinr'][0])
            user_sinr2 = [[] for _ in range(user_num)]
            # user_mcs = [[] for _ in range(user_num)]
            # user_mcs_ideal = [[] for _ in range(user_num)]
            postsinr_estimation2 = [[] for _ in range(user_num)]
            postsinr_estimation_raw2 = [[] for _ in range(user_num)]
            # user_gain_ave2 = [[] for _ in range(user_num)]
            # user_interference_ave2 = [[] for _ in range(user_num)]
            # user_interference_plus_noise_ave2 = [[] for _ in range(user_num)]
            # gain_estimation_ave2 = [[] for _ in range(user_num)]
            # interference_estimation_ave2 = [[] for _ in range(user_num)]
            for u in range(user_num):
                user_sinr2[u] = [sinr[u] for sinr in data_dict['user_sinr']]
                # user_mcs[u] = [mcs[u] for mcs in data_dict['user_mcs']]
                # user_mcs_ideal[u] = [mcs[u] for mcs in data_dict['user_mcs_ideal']]
                postsinr_estimation2[u] = [postsinr[u] for postsinr in data_dict['postsinr_estimation']]
                postsinr_estimation_raw2[u] = [postsinr_raw[u] for postsinr_raw in data_dict['postsinr_estimation_raw']]
                # user_gain_ave2[u] = [user_gain_ave[u] for user_gain_ave in data_dict['user_gain_ave']]
                # user_interference_ave2[u] = [user_interference_ave[u] for user_interference_ave in data_dict['user_interference_ave']]
                # user_interference_plus_noise_ave2[u] = [user_interference_plus_noise_ave[u] for user_interference_plus_noise_ave in data_dict['user_interference_plus_noise_ave']]
                # gain_estimation_ave2[u] = [gain_estimation_ave[u] for gain_estimation_ave in data_dict['gain_estimation_ave']]
                # interference_estimation_ave2[u] = [interference_estimation_ave[u] for interference_estimation_ave in
                #                              data_dict['interference_estimation_ave']]

                plt.figure()
                plt.plot(user_sinr2[u], label="real sinr")
                plt.plot(postsinr_estimation2[u], label="estimated sinr")
                plt.plot(postsinr_estimation_raw2[u], label="estimated sinr (without OLLA)")
                # plt.plot(user_mcs[u], label="mcs")
                plt.grid(True)
                plt.legend()
                # plt.ylim(-10,15)
                plt.title(f"sinr condition of user {u}")
                plt.show()

                filename = f"sinr condition of user {u}.png"
                filepath = os.path.join(pic_save_path, filename)
                plt.savefig(filepath)

                # plt.figure()
                # for layer in range(self._cfgs.env_cfgs.N_layer):
                #     plt.plot([ga[layer] for ga in user_gain_ave2[u]], label=f"ave gain layer {layer}")
                #     plt.plot([gea[layer] for gea in gain_estimation_ave2[u]], label=f"ave gain estimation layer {layer}")
                # # plt.plot(user_mcs[u], label="mcs")
                # plt.grid(True)
                # plt.legend()
                # # plt.ylim(-10,15)
                # plt.title(f"ave gain condition of user {u}")
                #
                # filename = f"ave gain condition of user {u}.png"
                # filepath = os.path.join(pic_save_path, filename)
                # plt.savefig(filepath)
                #
                # plt.show()
                #
                # plt.figure()
                # for layer in range(self._cfgs.env_cfgs.N_layer):
                #     plt.plot([ia[layer] for ia in user_interference_ave2[u]], label=f"ave interference layer {layer}")
                #     plt.plot([iea[layer] for iea in interference_estimation_ave2[u]], label=f"ave interference estimation layer {layer}")
                # # plt.plot(user_mcs[u], label="mcs")
                # plt.grid(True)
                # plt.legend()
                # # plt.ylim(-10,15)
                # plt.title(f"ave interference condition of user {u}")
                #
                # filename = f"ave interference condition of user {u}.png"
                # filepath = os.path.join(pic_save_path, filename)
                # plt.savefig(filepath)
                #
                # plt.show()
                #
                # plt.figure()
                # for layer in range(self._cfgs.env_cfgs.N_layer):
                #     plt.plot([nia[layer] for nia in user_interference_plus_noise_ave2[u]], label=f"ave interference + noise layer {layer}")
                # # plt.plot(user_mcs[u], label="mcs")
                # plt.grid(True)
                # plt.legend()
                # # plt.ylim(-10,15)
                # plt.title(f"ave interference + noise condition of user {u}")
                #
                # filename = f"ave interference + noise condition of user {u}.png"
                # filepath = os.path.join(pic_save_path, filename)
                # plt.savefig(filepath)
                #
                # plt.show()

        return data_dict


if __name__ == '__main__':
    T = 160*10

    # 验证所提算法训练模型
    save = \
            "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/多小区/QuaDRiGa/天花板"
    cfgs = get_default_kwargs_yaml('P3O')
    eval_obj = Evaluator(cfgs, save)
    data_dict = eval_obj.evaluate(T, need_plot=True)

    data_file_name = "eval_data.json"
    data_file_path = os.path.join(save, data_file_name)

    os.makedirs(save, exist_ok=True)

    # 保存为JSON
    with open(data_file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)  # indent=4 格式化输出，增强可读性

    # # 读取JSON文件
    # with open(data_file_path, "r", encoding="utf-8") as f:
    #     loaded_dict = json.load(f)