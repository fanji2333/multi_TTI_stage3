# import torch
import numpy as np

from common.tools import get_dis, determine_rows_cols

from omnisafe.utils.config import Config
# from omnisafe.common.normalizer import Normalizer

import os
from omnisafe.utils.tools import get_device
from omnisafe.utils.config import Config
from common.tools import plot, plot_bar
from common.tools import get_default_kwargs_yaml
from common.tools import BiasCorrectedEWMA


class User:
    def __init__(self, id, pos, BLER_T):
        self.id = id
        self.pos = pos
        self.BLER_T = BLER_T
        self.serve_BS: BS = None
        self.BLER = 0

    def reset(self):
        self.BLER = 0

    def update_BLER(self, ack, slot):
        self.BLER = (self.BLER * (slot - 1) + (1 - ack)) / slot

class BS:
    def __init__(self, id, P, noise, pos, max_UE, M, N_rf, SRS_period, buffer_len, codebook, rho):
        self.id = id
        self.pos = pos
        self.P = P
        self.noise = noise
        self.max_UE = max_UE
        self.M = M
        self.N_rf = N_rf
        self.SRS_period = SRS_period
        self.buffer_len = buffer_len
        self.codebook = codebook
        self.rho = rho
        self.rho2 = 20 * np.log10(rho)
        self.R = None
        self.serve_UEs = []
        self.ACK_dict = {}
        self.OLLA = {}
        self.OLLA_max = 10
        self.OLLA_min = -20
        self.OLLA_step = 2
        self.H_bs_total: np.ndarray = None
        self.H_bs_serve: np.ndarray = None
        self.H_l_bs_total: np.ndarray = None
        self.H_l_bs_serve: np.ndarray = None
        self.outer_precoder = None
        self.inner_precoder = None
        self.precoder = None
        self.CSI_update_delay = 0
        self.H_bs_correlation = []

    def collect_channels(self, H, Hl, slots, R):
        if slots % self.SRS_period == 0:
            self.H_bs_total = H[:, self.id, :]
            self.H_l_bs_total = Hl[:, self.id, 0]
            serve_UEs_ids = [u.id for u in self.serve_UEs]
            self.H_bs_serve = self.H_bs_total[serve_UEs_ids, :]
            self.H_l_bs_serve = self.H_l_bs_total[serve_UEs_ids]
            self.CSI_update_delay = 0
            self.R = R
            # correlation_matrix = self.H_bs_serve[0, :].reshape(1, -1).conj().T @ self.H_bs_serve[0, :].reshape(1, -1)
            # for i in range(0, len(serve_UEs_ids)):
            #     correlation_matrix = self.H_bs_serve[i, :].reshape(1,-1).T @ self.H_bs_serve[i, :].reshape(1, -1).conj()
            #     self.H_bs_correlation[i].update(correlation_matrix / (self.H_l_bs_serve[i]**2))
        else:
            self.CSI_update_delay += 1

        # print(f"距离上次更新CSI已经过去{self.CSI_update_delay}")

    # def generate_outer_precoder(self):
    #     # baseline直接选择最匹配的码字
    #     self.outer_precoder = np.zeros((self.M, self.N_rf), dtype=complex)
    #     for i, u in enumerate(self.serve_UEs):
    #         idx = np.argmax(self.H_bs_serve[u.id, :] @ self.codebook)
    #         self.outer_precoder[:, i] = self.codebook[:, idx]

    # def generate_outer_precoder(self):
    #     # TODO: 直接选择最匹配的前若干个码字
    #     self.outer_precoder = np.zeros((self.M, self.N_rf), dtype=complex)
    #     N = self.N_rf // len(self.serve_UEs)
    #     used_idx = []
    #     for i, u in enumerate(self.serve_UEs):
    #         sorted_idx = np.argsort(self.H_bs_serve[u.id, :] @ self.codebook)[::-1]
    #         # sorted_idx = sorted_idx[:N]
    #         for j in range(N):
    #             k = j
    #             while sorted_idx[k] in used_idx:
    #                 k += 1
    #             self.outer_precoder[:, i*N+j] = self.codebook[:, sorted_idx[k]]
    #             used_idx.append(sorted_idx[k])
    #     print(f"used outer precoder idx are {used_idx}")

    # def generate_outer_precoder(self):
    #     # 用信道自相关矩阵生成外层预编码
    #     eigenvalues, eigenvectors = np.linalg.eig(self.H_bs_correlation.corrected_avg)
    #
    #     # 获取特征值从大到小的索引
    #     sorted_indices = np.argsort(eigenvalues.real)[::-1]
    #
    #     # 按特征值大小排序
    #     sorted_eigenvalues = eigenvalues[sorted_indices]
    #     sorted_eigenvectors = eigenvectors[:, sorted_indices]
    #
    #     rank = np.sum(np.abs(sorted_eigenvalues) > 1e-20)
    #
    #     M = min(rank, self.N_rf)
    #     V = sorted_eigenvectors[:, :M]
    #     A = np.random.randn(self.N_rf, M) + 1j * np.random.randn(self.N_rf, M)
    #     U, _ = np.linalg.qr(A)
    #     # test_1 = U.conj().T @ U
    #     # test_2 = V.conj().T @ V
    #     self.outer_precoder = V @ U.conj().T

    def generate_outer_precoder(self):
        # 用信道自相关矩阵生成外层预编码，参考JSDM思路
        B = []
        for u in self.serve_UEs:
            eigenvalues, eigenvectors = np.linalg.eigh(self.R[u.id])

            # 获取特征值从大到小的索引
            sorted_indices = np.argsort(eigenvalues.real)[::-1]

            # 按特征值大小排序
            sorted_eigenvalues = eigenvalues[sorted_indices]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            rank = np.sum(np.abs(sorted_eigenvalues) > 1e-6)

            # M = min(rank, self.N_rf)
            V = sorted_eigenvectors[:, :rank]

            test1 = self.H_bs_serve[u.id, :] @ sorted_eigenvectors
            test2 = self.H_bs_serve[u.id, :] @ V
            B.append(V)
        self.outer_precoder = B

    # def generate_inner_precoder(self):
    #     H_RF: np.ndarray = self.H_bs_serve @ self.outer_precoder
    #     K = len(self.serve_UEs)
    #     temp = np.abs(H_RF)
    #     assert H_RF.shape == (K, self.N_rf), "RF chain channel dimension mismatch!"
    #     self.inner_precoder = H_RF.conj().T @ np.linalg.inv(H_RF @ H_RF.conj().T + K * self.noise / self.P * np.eye(K))

    def generate_inner_precoder(self):
        # 适配类JSDM算法，这里认为外层已经消除了用户间干扰，因此内层直接MRT
        W = []
        for u in self.serve_UEs:
            H_RF: np.ndarray = self.H_bs_serve[u.id, :].conj() @ self.outer_precoder[u.id]
            W.append(H_RF.conj().T)
        self.inner_precoder = W

    def generate_precoder(self, double_precoder: bool):
        K = len(self.serve_UEs)
        if double_precoder:
            self.generate_outer_precoder()
            self.generate_inner_precoder()
            # self.precoder = self.outer_precoder @ self.inner_precoder
            self.precoder = np.zeros((self.M, K), dtype=np.complex128)
            for u in self.serve_UEs:
                self.precoder[:, u.id] = self.outer_precoder[u.id] @ self.inner_precoder[u.id]
        else:
            # MMSE
            # inv = np.linalg.inv(self.H_bs_serve.conj() @ self.H_bs_serve.T + K * self.noise / self.P * np.eye(K))
            # self.precoder = self.H_bs_serve.T @ inv
            # MRT
            self.precoder = self.H_bs_serve.T
            # noise_matrix = np.eye(K) * (1 - self.rho ** (2*self.CSI_update_delay)) * K * self.P
            # for i, u in enumerate(self.serve_UEs):
            #     noise_matrix[i][i] *= self.H_l_bs_serve[i] ** 2
            # noise_matrix += self.noise * np.eye(K)
            # self.precoder = self.rho ** (self.CSI_update_delay) * self.H_bs_serve.conj().T @ np.linalg.inv(
            #     self.rho ** (2 * self.CSI_update_delay) * self.H_bs_serve @ self.H_bs_serve.conj().T + K / self.P * noise_matrix)
            # self.precoder = self.H_bs_serve.conj().T @ np.linalg.inv(
            #     self.H_bs_serve @ self.H_bs_serve.conj().T + K / self.P * noise_matrix)
        self.precoder = np.sqrt(self.P / np.trace(self.precoder.conj().T @ self.precoder)) * self.precoder

    def choose_mcs(self, mcs_table: dict, double_precoder: bool, ceiling: bool):
        # if double_precoder:
        #     H_RF: np.ndarray = self.H_bs_serve @ self.outer_precoder
        mcs_list = []
        postSINR_estimation_list = []
        for u in self.serve_UEs:
            if not ceiling:
                # TODO: SINR估计算法可优化
                if double_precoder:
                    # gain = np.linalg.norm(H_RF[u.id, :]) ** 2
                    # mu_loss = np.sum(
                    #     [(np.abs(H_RF[v.id, :] @ H_RF[u.id, :].conj().T) ** 2) / gain for v in self.serve_UEs]) - gain
                    gain = np.linalg.norm(self.H_bs_serve[u.id, :].conj() @ self.outer_precoder[u.id]) ** 2
                    # gain = gain / self.outer_precoder[u.id].shape[1]
                    mu_loss = 0
                else:
                    gain = np.linalg.norm(self.H_bs_serve[u.id, :]) ** 2
                    mu_loss = np.sum(
                        [(np.abs(self.H_bs_serve[v.id, :] @ self.H_bs_serve[u.id, :].conj().T) ** 2) / gain for v in self.serve_UEs]) - gain

                sinr_estimate = 10 * np.log10(gain / (mu_loss + self.noise / (self.P/len(self.serve_UEs)))) + self.OLLA[u.id]
            else:
                up = 0
                down = 0

                for v in self.serve_UEs:
                    precoding_vector = self.precoder[:, self.get_user_idx(v)]

                    if v == u:
                        up = np.linalg.norm(self.H_bs_serve[u.id, :].conj() @ precoding_vector) ** 2
                    else:
                        down += np.linalg.norm(self.H_bs_serve[u.id, :].conj() @ precoding_vector) ** 2

                sinr_estimate = 10 * np.log10(up / (down + self.noise)) + self.OLLA[u.id]
                # print(f"sinr_estimate is {sinr_estimate}")
            postSINR_estimation_list.append(sinr_estimate)
            mcs = "1"
            for key, value in mcs_table.items():
                if sinr_estimate >= value[1]:
                    mcs = key
                else:
                    break
            mcs_list.append(mcs)
        return mcs_list, postSINR_estimation_list

    def get_user_idx(self, u: User):
        assert u in self.serve_UEs, "User not served!"
        return self.serve_UEs.index(u)

    def log_user(self, u: User):
        self.serve_UEs.append(u)
        self.ACK_dict[u.id] = [1 for _ in range(self.buffer_len)]
        self.OLLA[u.id] = 0
        self.H_bs_correlation.append(BiasCorrectedEWMA(alpha=0.3))

    def update_ACK(self, u: User, ack, OLLA_scheme):
        self.ACK_dict[u.id].append(ack)
        self.ACK_dict[u.id] = self.ACK_dict[u.id][1:]
        assert len(self.ACK_dict[u.id]) == self.buffer_len, "ACK list length is not equal to buffer_len!"
        if OLLA_scheme:
            # TODO: OLLA周期重置
            if self.CSI_update_delay == 159:
                self.OLLA[u.id] = 0
            else:
                if ack:
                    self.OLLA[u.id] += self.OLLA_step * u.BLER_T
                    if self.OLLA[u.id] >= self.OLLA_max:
                        self.OLLA[u.id] = self.OLLA_max
                else:
                    self.OLLA[u.id] -= self.OLLA_step * (1 - u.BLER_T)
                    if self.OLLA[u.id] <= self.OLLA_min:
                        self.OLLA[u.id] = self.OLLA_min
        else:
            if self.CSI_update_delay == 159:
                self.OLLA[u.id] = 0
            else:
                self.OLLA[u.id] += self.rho2

    def reset(self):
        self.serve_UEs = []
        self.ACK_dict = {}
        self.H_bs_total = None
        self.H_bs_serve = None
        self.H_l_bs_total = None
        self.H_l_bs_serve = None
        self.outer_precoder = None
        self.inner_precoder = None
        self.CSI_update_delay = 0
        for H_bs_correlation in self.H_bs_correlation:
            H_bs_correlation.reset()

class Environment:

    def __init__(self, cfg: Config, device):

        self._cfg = cfg
        self._device = device

        # 环境基本设置
        self._B = cfg.B              # 基站数量
        self._K = cfg.U              # 用户数量
        self._M = cfg.M              # BS天线数量
        self._Bw = cfg.Bw            # 单个子载波带宽(kHz)
        self._N = cfg.N              # 每个时隙的OFDM符号数

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
        self._sigma = self._noise * self._Bw * 1e3          # 噪声功率(W)

        self._OFDM_t = 1 / (self._Bw * 1e3)                 # OFDM符号时长
        self._slot_t = cfg.t_slot                           # 时隙时长(ms)
        # self._f_carrier = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 for n in range(self._F)]   # 各载波频率(MHz)
        # self._f_subband = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 * self._C for n in range(self._K)]   # 各子带中心频率(MHz)
        self._H = None                                      # 信道矩阵
        self._Hl = None                                     # 信道大尺度衰落
        self._Hs = None                                     # 信道小尺度衰落
        self._P = 10 ** (cfg.P/10 - 3)                                   # 固定发射功率分配(W)

        # # 测试有干扰情况下的sinr分布情况
        # # 用户数*用户数，第一个参数表明是哪个用户，第二个参数表明存在多少用户干扰
        # self.sinr_interference_condition = [[[] for _ in range(self._K)] for _ in range(self._K)]

        # 用户设置参数
        Users_pos = cfg.Users_pos                      # 用户位置
        self._h_U = cfg.h_U                            # 用户高度
        self._BLER_T = cfg.BLER_T                      # 用户BLER阈值
        # shadow_user = np.random.normal(0, 1, size=self._K)     # User侧的shadow fading参数

        # BS设置参数
        # self._max_P = cfg.max_P            # 每个BS所能承受的最大功率单元数
        # self._dP = 10 ** (cfg.max_P_dBm / 10 - 3) / cfg.max_P           # 功率分配粒度(W)
        self._h_BS = cfg.h_BS              # BS高度
        self._K_BS = cfg.K_BS              # BS最大连接用户数
        self._N_rf = cfg.N_rf              # BS的RF chain数
        self._SRS_period = cfg.SRS_period  # SRS更新周期
        self._buffer_len = cfg.buffer_len  # BS存储历史数据条数
        # shadow_BS = np.random.normal(0, 1, size=self._B)       # BS侧的shadow fading参数

        # self._Users = cfg["Users"]      # 用户数据
        # self._BSs = cfg["BSs"]          # BS数据

        self._region_bound = cfg.region_bound                # 所考虑区域大小
        self._min_distance_BS = cfg.min_distance_BS          # BS间最小间距
        self._min_distance_User = cfg.min_distance_User      # 用户间最小间距

        self.MCS_table = {
            # MCS阶数: [编码效率(bits/symbol), SINR阈值(dB)]
            # SINR大于阈值则可以选择对应MCS，此时最优MCS为SINR恰大于本阶阈值而小于下阶阈值
            '1': [0.15, -6.50],
            '2': [0.23, -4.00],
            '3': [0.38, -2.60],
            '4': [0.60, -1.00],
            '5': [0.88, 1.00],
            '6': [1.18, 3.00],
            '7': [1.48, 6.60],
            '8': [1.91, 10.00],
            '9': [2.41, 11.40],
            '10': [2.73, 11.80],
            '11': [3.32, 13.00],
            '12': [3.90, 13.80],
            '13': [4.52, 15.60],
            '14': [5.12, 16.80],
            '15': [5.55, 17.60],
        }

        # 生成用户
        self.UEs: list[User] = []
        for i in range(self._K):
            self.UEs.append(User(i, Users_pos[i], self._BLER_T))

        # 设置BS位置
        BS_pos_cfg = determine_rows_cols(self._B, self._region_bound, self._region_bound)
        if BS_pos_cfg == (1, 1):
            BS_pos = [(self._region_bound / 2, self._region_bound / 2)]
        else:
            # 计算行间距和列间距
            dx = self._region_bound / (BS_pos_cfg[1] - 1)
            dy = self._region_bound / (BS_pos_cfg[0] - 1)
            # 生成点的坐标
            BS_pos = [(dx * j, dy * i) for i in range(BS_pos_cfg[0]) for j in range(BS_pos_cfg[1])]

        # 设置是否双层预编码
        self._double_precoder = cfg.double_precoder
        # 是否L2获取L1实际预编码
        self._ceiling = cfg.ceiling
        # 是否传统OLLA机制
        self._OLLA_scheme = cfg.OLLA

        # 生成DFT码本
        # self._DFT_codebook = 1 / np.sqrt(self._M) * np.exp(
        #     2j * np.pi * np.outer(np.arange(self._M), np.arange(self._M)) / self._M)
        # 两倍天线量码本
        # self._DFT_codebook = 1 / np.sqrt(self._M) * np.exp(
        #     2j * np.pi * np.outer(np.arange(self._M*2), np.arange(self._M*2)) / (self._M*2))
        # self._DFT_codebook = self._DFT_codebook[0:self._M, :]
        # 四倍天线量码本
        # self._DFT_codebook = 1 / np.sqrt(self._M) * np.exp(
        #     2j * np.pi * np.outer(np.arange(self._M*4), np.arange(self._M*4)) / (self._M*4))
        # self._DFT_codebook = self._DFT_codebook[0:self._M, :]
        # # 八倍天线量码本
        # self._DFT_codebook = 1 / np.sqrt(self._M) * np.exp(
        #     2j * np.pi * np.outer(np.arange(self._M * 8), np.arange(self._M * 8)) / (self._M * 8))
        # self._DFT_codebook = self._DFT_codebook[0:self._M, :]
        # 尝试码字合成来加宽
        self._DFT_codebook_temp = 1 / np.sqrt(self._M) * np.exp(
            2j * np.pi * np.outer(np.arange(self._M), (np.arange(self._M) + (1-self._M)/2)) / self._M)
        self._DFT_codebook = np.zeros((self._M, self._M-1), dtype=np.complex128)
        for i in range(self._M-1):
            self._DFT_codebook[:, i] = self._DFT_codebook_temp[:, i] * np.exp(1j * np.pi * (-1 + 1/self._M) * (i+1)) + self._DFT_codebook_temp[:, i+1] * np.exp(1j * np.pi * (-1 + 1/self._M) * (i+2))

        # # 生成信道空间相关矩阵
        # self._R = np.zeros((self._M, self._M))
        # for i in range(self._M):
        #     for j in range(self._M):
        #         self._R[i][j] = self._r ** np.abs(i - j)

        # 生成BS
        self.BSs: list[BS] = []
        for i in range(self._B):
            # temp = {"max P": self._max_P, "location": BS_pos[i], "shadow": shadow_BS[i]}
            self.BSs.append(BS(i, self._P, self._sigma, BS_pos[i], self._K_BS, self._M, self._N_rf,
                               self._SRS_period, self._buffer_len, self._DFT_codebook, self._rho))

        # 生成信道空间相关矩阵
        self._R = np.zeros((self._K, self._M, self._M), dtype=np.complex128)
        for u in self.UEs:
            for m in range(self._M):
                for n in range(self._M):
                    theta_bar = self.calculate_AOD(self.BSs[0].pos[0], self.BSs[0].pos[1], u.pos[0], u.pos[1])
                    self._R[u.id][m][n] = 1 / self._M * np.exp(
                        -1j * self._kai * (m - n) * np.cos(theta_bar)) * np.sinc(
                        self._kai * (m - n) * self._delta_theta * np.sin(theta_bar))

        # 记录用户MCS选择分布情况
        self.user_MCS_distribution = []
        for i in range(self._K):
            self.user_MCS_distribution.append([0] * 15)

        self._slots = 0

        self._tot_bits = 0
        self.state = None

        # # print(f"obs dimension is {self.get_obs_dim()}")
        # # 设置normalizer
        # if self._cfg.obs_normalize:
        #     self._obs_normalizer = Normalizer((self.get_obs_dim(),), clip=25).to(self._device)
        # if self._cfg.reward_normalize:
        #     self._reward_normalizer = Normalizer((), clip=5).to(self._device)
        # if self._cfg.cost_normalize:
        #     self._cost_normalizer = Normalizer((self.get_cost_num(),), clip=25).to(self._device)

    def reset(self):

        self._slots = 0
        self._tot_bits = 0

        # 初始化信道大尺度衰落情况
        # 暂时不考虑用户位置的变化，认为大尺度衰落特性不变
        self._Hl = np.zeros((self._K, self._B, self._M))
        for k in range(self._K):
            for b in range(self._B):
                dub = get_dis(self.UEs[k].pos, self.BSs[b].pos, self._h_U, self._h_BS)
                self._Hl[k, b, :] = np.sqrt(self.get_largescale(k, b, dub, self._fc))

        # 初始化信道小尺度衰落情况
        self._Hs = self.generate_gaussian_channel(self._R)

        # 得到信道矩阵
        self._H = self._Hs * self._Hl

        for bs in self.BSs:
            bs.reset()

        # 设置BS与AP关联
        # TODO: 二阶段只要求单基站，这里直接单基站处理
        for u in self.UEs:
            self.BSs[0].log_user(u)
            u.serve_BS = self.BSs[0]

        # 各个BS整理信道信息
        # 只有SRS周期时BS才会更新信道
        for bs in self.BSs:
            bs.collect_channels(self._H, self._Hl, self._slots, self._R)

        # # 将要返回给agent的状态打包
        # # TODO: 二阶段只要求单基站，这里直接单基站处理
        # tensor_H_real = torch.from_numpy(self.BSs[0].H_bs_serve.real).flatten()
        # tensor_H_imag = torch.from_numpy(self.BSs[0].H_bs_serve.imag).flatten()
        # ACK_list_temp = []
        # for UE_id, ack_list in self.BSs[0].ACK_dict.items():
        #     ACK_list_temp.append(ack_list)
        # tensor_ACK_list = torch.tensor(ACK_list_temp).flatten()
        # tensor_update_delay = torch.Tensor([self.BSs[0].CSI_update_delay])
        #
        # self.state = torch.cat((tensor_H_real, tensor_H_imag,
        #                         tensor_ACK_list, tensor_update_delay), dim=0).float().to(self._device)
        info = {}

        # state = self.state
        state = None

        # TODO: obs是否需要归一化？
        # if self._cfg.obs_normalize:
        #     origin_state = self.state
        #     state = self._obs_normalizer.normalize(self.state)
        #     info['origin_state'] = origin_state

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

        # TODO: 二阶段只要求单基站，这里直接单基站处理
        # 计算内层数字预编码
        for bs in self.BSs:
            # bs.generate_outer_precoder()
            # bs.generate_inner_precoder()
            bs.generate_precoder(self._double_precoder)
            MCS_list, postSINR_estimation_list = bs.choose_mcs(self.MCS_table, self._double_precoder, self._ceiling)

        # print(f"MCS_list: {MCS_list}")

        # 根据调度结果更新用户数据队列，记录这一时隙内总发送比特数，并考察时延约束违反情况
        tot_bits = 0
        success_users = self._K
        ACK_list = []
        user_bits = []
        user_BLER = []
        user_OLLA = []
        user_sinr = []
        for i, u in enumerate(self.UEs):

            # TODO: 二阶段只要求单基站因此这样处理，如果三阶段扩展到多基站则还需要辨析用户是否受到agent指导
            bits, ACK, info = self.get_rate(u, str(MCS_list[i]))
            u.serve_BS.update_ACK(u, ACK, self._OLLA_scheme)
            u.update_BLER(ACK, self._slots)

            # 记录各类参数
            tot_bits += bits
            user_bits.append(bits)
            ACK_list.append(ACK)
            user_BLER.append(u.BLER)
            user_OLLA.append(u.serve_BS.OLLA[u.id])
            user_sinr.append(info["sinr"])
            self.user_MCS_distribution[i][int(MCS_list[i]) - 1] += 1

        # tot_bps = tot_bits / (self._slot_t * 1e-3)
        # user_bps = [ele / (self._slot_t * 1e-3) for ele in user_bits]

        # 奖励
        # 直接以总传输bit数为奖励
        reward = tot_bits / 10

        # 成本
        cost_list = []
        # 直接用瞬时ACK/NACK情况结合BLER目标计算cost
        for i, u in enumerate(self.UEs):
            cost_u = (1 - ACK_list[i]) - u.BLER_T
            cost_list.append(cost_u)

        # 本时隙结束，下一时隙开始，随机化信道小尺度衰落情况
        # 生成小尺度衰落改变量
        G = self.generate_gaussian_channel(self._R)
        # 更新小尺度衰落
        self._Hs = self._rho * self._Hs + self._rho2 * G
        # 得到信道矩阵
        self._H = self._Hs * self._Hl

        # 各个BS整理信道信息
        # 只有SRS周期时BS才会更新信道
        for bs in self.BSs:
            bs.collect_channels(self._H, self._Hl, self._slots, self._R)

        # # 将要返回给agent的状态打包
        # # TODO: 二阶段只要求单基站，这里直接单基站处理
        # tensor_H_real = torch.from_numpy(self.BSs[0].H_bs_serve.real).flatten()
        # tensor_H_imag = torch.from_numpy(self.BSs[0].H_bs_serve.imag).flatten()
        # ACK_list_temp = []
        # for UE_id, ack_list in self.BSs[0].ACK_dict.items():
        #     ACK_list_temp.append(ack_list)
        # tensor_ACK_list = torch.tensor(ACK_list_temp).flatten()
        # tensor_update_delay = torch.Tensor([self.BSs[0].CSI_update_delay])
        #
        # self.state = torch.cat((tensor_H_real, tensor_H_imag,
        #                         tensor_ACK_list, tensor_update_delay), dim=0).float().to(self._device)

        info = {
            'tot_bits': tot_bits,
            'cost_list': cost_list,
            'user_bits': user_bits,
            'user_BLER': user_BLER,
            'user_OLLA': user_OLLA,
            'user_sinr': user_sinr,
            'postsinr_estimation': postSINR_estimation_list,
            'user_MCS_distribution': self.user_MCS_distribution,
        }

        # state = self.state
        #
        # # if self._cfg.obs_normalize:
        # #     origin_state = self.state
        # #     state = self._obs_normalizer.normalize(self.state)
        # #     info['origin_state'] = origin_state
        # # else:
        # #     info['origin_state'] = self.state
        #
        # assert not torch.isnan(state).any(), "state contains NaN values"
        #
        # reward = torch.tensor(reward).to(self._device)
        # # if self._cfg.reward_normalize:
        # #     origin_reward = reward
        # #     reward = self._reward_normalizer.normalize(reward.float())
        # #     info['origin_reward'] = origin_reward
        # # else:
        # #     info['origin_reward'] = reward
        # #
        # cost = torch.tensor(cost_list).to(self._device)
        # # if self._cfg.cost_normalize:
        # #     origin_cost = cost
        # #     cost = self._cost_normalizer.normalize(cost.float())
        # #     info['origin_cost'] = origin_cost
        # # else:
        # #     info['origin_cost'] = cost
        # info['origin_reward'] = reward
        # info['origin_cost'] = cost

        state, cost = None, None

        # 返回内容：输入给agent的状态、奖励、成本、性能指标
        return state, reward, cost, info

    # def save(self):
    #
    #     save = {}
    #     if self._cfg.obs_normalize:
    #         save['obs_normalizer'] = self._obs_normalizer
    #     if self._cfg.reward_normalize:
    #         save['reward_normalizer'] = self._reward_normalizer
    #     if self._cfg.cost_normalize:
    #         save['cost_normalizer'] = self._cost_normalizer
    #
    #     return save
    #
    # def load_norm(self, **params):
    #
    #     if ('obs_norm_param' in params) and self._cfg.obs_normalize:
    #         self._obs_normalizer.load_state_dict(params['obs_norm_param'])
    #     if ('reward_norm_param' in params) and self._cfg.reward_normalize:
    #         self._reward_normalizer.load_state_dict(params['reward_norm_param'])
    #     if ('cost_norm_param' in params) and self._cfg.cost_normalize:
    #         self._cost_normalizer.load_state_dict(params['cost_norm_param'])

    # def process_action(self, action: torch.Tensor):
    #     K = len(self.BSs[0].serve_UEs)
    #     mcs = action[0:K].tolist()
    #     precoder_idx = action[K:].tolist()
    #     assert len(mcs) == K, "mcs dimension mismatch!"
    #     assert len(precoder_idx) == self._N_rf, "outer precoder dimension mismatch!"
    #
    #     mcs = self.linear_map(mcs, (1, 15), (-4, 4))
    #     precoder_idx = self.linear_map(precoder_idx, (0, self._M - 1), (-8, 8))
    #
    #     return mcs, precoder_idx

    # def bounded_round(self, x, lower_bound, upper_bound):
    #     """
    #     将输入数据四舍五入到最近的整数，并有上下界限制
    #
    #     Args:
    #         x: 输入数据，可以是标量、列表、numpy数组等
    #         lower_bound: 下界，小于此值的数会被舍入到下界
    #         upper_bound: 上界，大于此值的数会被舍入到上界
    #
    #     Returns:
    #         四舍五入并限制边界后的结果
    #     """
    #     # 首先进行四舍五入
    #     rounded = np.round(x)
    #
    #     # 应用下界限制
    #     rounded = np.maximum(rounded, lower_bound)
    #
    #     # 应用上界限制
    #     rounded = np.minimum(rounded, upper_bound)
    #
    #     return rounded.astype(int)

    def linear_map(
            self,
            input_list: list[float],
            target_bounds: tuple[int, int],
            input_bounds: tuple[float, float]
    ) -> list[int]:
        """
        将输入列表中的数字线性映射到目标整数区间

        Args:
            input_array: 包含数字的输入列表
            target_bounds: 目标整数上下界 (min, max)
            input_bounds: 输入数字上下界 (min, max)

        Returns:
            映射到目标整数区间的列表
        """
        # 解包边界值
        target_min, target_max = target_bounds
        input_min, input_max = input_bounds

        # 验证边界有效性
        assert input_min < input_max, "输入下界必须小于输入上界"
        assert target_min < target_max, "目标下界必须小于目标上界"

        input_array = np.array(input_list)

        # 第一步：将输入值限制在输入边界内
        clipped_array = np.clip(input_array, input_min, input_max)

        # 第二步：线性映射到目标区间
        # 公式: output = (input - input_min) * (target_range / input_range) + target_min
        input_range = input_max - input_min
        target_range = target_max - target_min

        # 执行线性映射并四舍五入到最近的整数
        mapped_array = (clipped_array - input_min) * (target_range / input_range) + target_min
        result_array = np.round(mapped_array).astype(int)

        # 确保结果在目标边界内（由于四舍五入可能超出边界）
        result_array = np.clip(result_array, target_min, target_max)

        return result_array.astype(int).tolist()

    # def generate_gaussian_channel(self, R):
    #     """
    #     高效批量生成复高斯随机向量
    #
    #     Args:
    #         R: 协方差矩阵 (M, M)
    #
    #     Returns:
    #         形状为 (batch_size, num_vectors, M) 的数组
    #     """
    #     M = R.shape[0]
    #     assert M == self._M, "shape of R mismatch with antenna num!"
    #     L = np.linalg.cholesky(R)
    #
    #     K = self._K
    #     B = self._B
    #
    #     # 一次性生成所有随机数
    #     total_samples = K * B
    #     h_real = np.random.randn(total_samples, M)
    #     h_imag = np.random.randn(total_samples, M)
    #     h = (h_real + 1j * h_imag) / np.sqrt(2)
    #
    #     # 批量变换
    #     h = (L @ h.T).T
    #
    #     # 重塑为批次格式
    #     return h.reshape(K, B, M)
    def generate_gaussian_channel(self, R):
        """
        高效批量生成复高斯随机向量

        Args:
            R: 协方差矩阵 (M, M)

        Returns:
            形状为 (batch_size, num_vectors, M) 的数组
        """

        # TODO: 这里只是单BS情况
        K = self._K
        B = self._B
        M = R.shape[1]
        assert M == self._M, "shape of R mismatch with antenna num!"
        h = np.zeros((K, B, M), dtype=np.complex128)

        for k in range(K):
            # 步骤1：特征值分解（对 Hermitian 矩阵用 eigh，确保数值稳定）
            eigenvalues, eigenvectors = np.linalg.eigh(R[k])

            # 步骤2：处理特征值（确保非负，避免数值误差导致的微小负数）
            epsilon = 1e-18
            eigenvalues = np.maximum(eigenvalues, epsilon)  # 替换负特征值为极小正数
            lambda_sqrt = np.sqrt(eigenvalues)  # 特征值开平方

            # 步骤3：构造平方根矩阵 L = U * Lambda^(1/2)
            L = eigenvectors @ np.diag(lambda_sqrt)

            # 一次性生成所有随机数
            total_samples = B
            h_real = np.random.randn(total_samples, M)
            h_imag = np.random.randn(total_samples, M)
            h0 = (h_real + 1j * h_imag) / np.sqrt(2)

            # 批量变换
            h[k, 0, :] = (L @ h0.T).T

        return h

    def calculate_AOD(self, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        theta_bar = np.arctan2(dx, dy)
        return np.abs(theta_bar)

    # 计算大尺度衰落
    def get_largescale(self, u, b, dub, fc):

        beta_dB = self._d0 + self._d1 * np.log10(dub) + self._d2 * np.log10(fc * 1e-3)

        return 10 ** (- beta_dB/10)

    # 计算用户SINR
    def get_SINR(self, u: User):
        up = 0
        down = 0
        H = self._H

        for v in self.UEs:

            for b in self.BSs:
                tx_energy = np.trace(b.precoder.conj().T @ b.precoder)
                if v in b.serve_UEs:
                    # # precoding_vector = b.outer_precoder @ b.inner_precoder[:, b.get_user_idx(v)]
                    # precoding_vector = b.precoder[:, b.get_user_idx(v)]
                    # # TODO:这里把内外层合并后再单独进行功率归一化
                    # precoding_vector = np.sqrt(self._P) / np.linalg.norm(precoding_vector) * precoding_vector
                    # precoder能量归一化在计算时已实现
                    precoding_vector = b.precoder[:, b.get_user_idx(v)]
                    # precoder_energy = np.linalg.norm(precoding_vector)**2
                    # ideal_gain = np.abs(b.H_bs_serve[v.id, :].conj() @ precoding_vector)**2
                    # ideal_gain2 = np.abs(b.H_bs_serve[v.id, :] @ b.H_bs_serve[v.id, :].conj().T)
                else:
                    continue

                if v == u:
                    up = np.linalg.norm(H[u.id, b.id, :].conj() @ precoding_vector) ** 2
                else:
                    down += np.linalg.norm(H[u.id, b.id, :].conj() @ precoding_vector) ** 2

        sinr = up / (down + self._sigma)
        # temp = 10 * np.log10(sinr)
        # ideal_sinr = 10 * np.log10(ideal_gain / self._sigma)

        # 用调度矩阵中非0行数量判断当前子带上是否有复用
        has_ICI = down

        return sinr, has_ICI

    # 计算某用户在时隙内的可发送比特数
    def get_rate(self, u: User, mcs: str):

        info = {}
        sinr, has_ICI = self.get_SINR(u)
        # if sinr != 0 and has_ICI:
        # if sinr != 0:
        # if sinr != 0:
        #     temp = 10 * np.log10(sinr)
        #     print(f"the SINR of user {u.id} is : {temp}")
            # self.sinr_interference_condition[u][has_ICI].append(temp)

        sinr = 10 * np.log10(sinr)
        # print(f"the SINR of user {u.id} is : {sinr}")
        if sinr >= self.MCS_table[mcs][1]:
            ACK = 1
            bits = self.MCS_table[mcs][0] * self._N
        else:
            ACK = 0
            bits = 0

        info["sinr"] = sinr
        # print(f"ACK/NACK: {ACK}")

        return bits, ACK, info
    #
    # def channel_compromize(self):
    #     H = self._H
    #
    #     for k in range(H.shape[0]):
    #         for u in range(H.shape[1]):
    #             for b in range(H.shape[2]):
    #                 # 暂时使用单基站MRT形式，多基站之间无协作
    #                 precoding_matrix = np.conj(H[k, u, b, :]) / np.linalg.norm(H[k, u, b, :])
    #                 self._H_C[k, u, b] = precoding_matrix @ H[k, u, b, :]

    # # 绘制点
    # def plot_points(self):
    #
    #     plt.rcParams['font.sans-serif'] = ['SimHei']
    #     plt.rcParams['axes.unicode_minus'] = False
    #
    #     BSs_location = [BS.location for BS in self._CPU.BSs]
    #     Users_location = [User.location for User in self._CPU.UEs]
    #
    #     plt.figure(figsize=(6, 6))
    #     plt.xlim((0, self._region_bound))
    #     plt.ylim((0, self._region_bound))
    #     plt.scatter(*zip(*BSs_location), color='pink', label='BSs')
    #     for i, u in enumerate(Users_location):
    #         plt.scatter(*u, label=f'User {i+1}')
    #     # plt.scatter(*zip(*Users_location), color='red', label='Users')
    #     plt.title('环境位置设置')
    #     plt.xlabel('X/m')
    #     plt.ylabel('Y/m')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    # 得到环境参数
    def get_obs_dim(self):
        # TODO: 二阶段只要求单基站，这里直接单基站处理
        # 状态为当前BS能获取的CSI、各用户历史ACK、CSI更新延迟
        return self._K_BS * self._M * 2 + self._K_BS * self._buffer_len + 1

    def get_act_dim(self):
        # TODO: 二阶段只要求单基站，这里直接单基站处理
        # 动作为每个用户的MCS选择以及外层波束选择
        return self._K_BS + self._N_rf

    def get_cost_num(self):
        return self._K


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
        user_OLLA = []
        user_sinr = []
        postsinr_estimation = []
        pic_save_path = os.path.dirname(os.path.dirname(self._save))
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
            user_OLLA.append(info['user_OLLA'])
            user_sinr.append(info['user_sinr'])
            postsinr_estimation.append(info['postsinr_estimation'])

            total_bits += info['tot_bits']

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
            "slots": slots,
            "slot_bits": slot_bits,
            "user_bits": user_bits,
        }

        if need_plot:
            # plot(step_costs, slots, 'costs per slot', pic_save_path)
            # plot(slot_bits, slots, 'total bits per slot', pic_save_path)
            # plot(user_bits, slots, 'bits per UE per slot', pic_save_path)
            plot(user_BLER, slots, 'BLER per UE per slot', pic_save_path)
            plot(user_OLLA, slots, 'OLLAs per UE per slot', pic_save_path)
            plot(user_sinr, slots, 'sinr per UE per slot', pic_save_path)
            plot(postsinr_estimation, slots, 'postsinr estimation per UE per slot', pic_save_path)
            plot(user_sinr_ave, range(160), 'ave sinr per UE per slot', pic_save_path)
            plot(postsinr_ave, range(160), 'ave postsinr estimation per UE per slot', pic_save_path)
            for i in range(len(user_MCS_distribution)):
                plot_bar([x / step_num for x in user_MCS_distribution[i]], None,
                         f'user {i} MCS distribution', "MCS order", pic_save_path)
            print(f'after slots {step_num}, average bits/slot is {total_bits / step_num}, BLER is about {user_BLER[-1]}')

        return data_dict


T = 160*1000

# 验证所提算法训练模型
save = \
        "/home/fj24/25_8_Huawei_multiTTI/runs/华为双层预编码/baseline_test"
cfgs = get_default_kwargs_yaml('PPO')
eval_obj = Evaluator(cfgs, save)
eval_obj.evaluate(T, need_plot=True)