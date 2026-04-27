import torch
import numpy as np
import scipy.special as sp
# import h5py

from common.tools import get_dis, determine_rows_cols, BiasCorrectedEWMA

from omnisafe.utils.config import Config
from omnisafe.common.normalizer import Normalizer

from scipy.io import loadmat
import os
from functools import lru_cache

'''
    本环境是为了提升QuaDriGa信道数据利用率而设置的。
    尽管方案中各个用户是并行执行传输，但由于智能体配置是单用户的，因此多用户并行传输的话数据集需要重新编排存储，会很麻烦，
    因此本环境每次只实际传输一个用户、采集一个用户的数据，另一用户跑完仿真的全流程，但不参与传输效果的计算及数据采集，
    允许外环训练调度具体采集哪个用户的数据，例如QuaDriGa仿真了两用户的信道，那么先只采集一个用户的数据并训练，
    等所有信道跑完后再回到开头重新跑，并且换成采集另一用户的数据，
    如此一来只用生成一组信道就能训练多轮，但也就需要运行多轮。
'''


def _scalar(x):
    import numpy as _np
    return int(_np.array(x).squeeze())

@lru_cache(maxsize=1)
def _load_meta(out_dir: str):
    m = loadmat(os.path.join(out_dir, "meta.mat"))
    meta = {}
    for k in ["PACK_SIZE", "NRB", "NBS", "N_RX", "Ns_per_UE", "U", "fc", "BW", "SCS"]:
        if k in m:
            meta[k] = _scalar(m[k])
    return meta

def _pack_path(out_dir: str, ue_id_1based: int, pack_id: int) -> str:
    return os.path.join(out_dir, f"UE{ue_id_1based:02d}_pack{pack_id:03d}.mat")

@lru_cache(maxsize=2)   # 小缓存，避免内存暴涨；需要可改大
def _load_pack(out_dir: str, ue_id_1based: int, pack_id: int):
    fn = _pack_path(out_dir, ue_id_1based, pack_id)
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"not found: {fn}")
    d = loadmat(fn, variable_names=["H_batch"])
    Hb = d["H_batch"]  # [K, NRB, NBS, N_RX]  complex single
    import numpy as _np
    if Hb.dtype != _np.complex64:
        Hb = Hb.astype(_np.complex64, copy=False)
    return Hb

class QuadrigaChannelSource:
    """按需读取：给定 ue_id(0-based) 和 tti(0-based)，返回 H[NRB,NBS,N_RX]"""
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.meta = _load_meta(out_dir)
        self.NRB = self.meta["NRB"]; self.NBS = self.meta["NBS"]; self.NRX = self.meta["N_RX"]
        self.PS  = self.meta["PACK_SIZE"]; self.Ns = self.meta["Ns_per_UE"]

    def get_H(self, ue_id_0based: int, tti_0based: int):
        # 文件名里的 UE 从 1 开始；MAT 索引从 1 开始，这里统一 0-based -> 1-based
        ue1 = ue_id_0based + 1
        tti1 = (tti_0based % self.Ns) + 1
        pack_id = (tti1 - 1) // self.PS + 1
        idx_in_pack = (tti1 - 1) % self.PS
        Hb = _load_pack(self.out_dir, ue1, pack_id)     # [K, NRB, NBS, N_RX]
        if idx_in_pack >= Hb.shape[0]:
            raise IndexError(f"TTI {tti1} not present in pack #{pack_id} (size {Hb.shape[0]})")
        H = Hb[idx_in_pack, :, :, :]                    # [NRB,NBS,N_RX]
        return H

class User:
    def __init__(self, id, BLER_T, Mr, N_layer):
        self.id = id
        self.BLER_T = BLER_T
        self.Mr = Mr
        self.n_layer = N_layer
        self.serve_BS: BS = None
        self.BLER = 0
        self.BLER_ideal = 0
        self.precoder = None
        self.combiner = None
        self.combiner_eff_gain = []

    def reset(self):
        self.BLER = 0
        self.BLER_ideal = 0
        self.precoder = None
        self.combiner = None
        self.combiner_eff_gain = []

    def update_BLER(self, ack, slot, ideal_bler=None):
        self.BLER = (self.BLER * (slot - 1) + (1 - ack)) / slot
        if ideal_bler is not None:
            self.BLER_ideal = (self.BLER_ideal * (slot - 1) + ideal_bler) / slot

class BS:
    def __init__(self, id, P, noise, max_UE, Mt, Mr, SRS_period, buffer_len, rho):
        self.id = id
        self.P = P
        self.noise = noise
        self.max_UE = max_UE
        self.Mt = Mt
        self.Mr = Mr
        self.SRS_period = SRS_period
        self.buffer_len = buffer_len
        # TODO: 暂时将rho取为试出来的固定值
        self.rho = [0.9966, 0.9966, 0.9966, 0.9966]
        # self.rho2 = 20 * np.log10(rho)
        self.Rt = {}
        self.Rr = {}
        self.serve_UEs = []
        self.ACK_dict = {}
        self.OLLA = {}
        self.OLLA_max = 25
        self.OLLA_min = 5
        self.OLLA_step = 2
        self.H_bs_total = {}
        self.H_bs_serve = {}
        self.H_l_bs_total = {}
        self.H_l_bs_serve = {}
        self.CSI_update_delay = 0
        self.P_user = {}
        self.H_bs_real = {}
        self.n_stream = 0
        self.WMMSE_max_iteration = 5
        self.mu_loss_per_user_dB = 2

    def collect_channels(self, H, slots):
        self.H_bs_real = H
        self.n_stream = 0
        for u in self.serve_UEs:
            self.n_stream += u.n_layer
        if slots % self.SRS_period == 0:
            self.H_bs_total = H
            for u in self.serve_UEs:
                self.H_l_bs_total[u.id], Hs = self.separate_large_scale_fading(H[u.id])
                self.Rr[u.id], self.Rt[u.id] = self.estimate_corr_matrix(Hs)
                eigenvalues_r, _ = np.linalg.eigh(self.Rr[u.id])
                eigenvalues_r = eigenvalues_r[::-1]
                u.combiner_eff_gain = eigenvalues_r[:u.n_layer]
                if slots != 0:
                    self.rho[u.id] = self.calculate_rho(self.H_bs_serve[u.id][0], self.H_bs_total[u.id][0])
                self.H_bs_serve[u.id] = self.H_bs_total[u.id]
                self.H_l_bs_serve[u.id] = self.H_l_bs_total[u.id]
            self.CSI_update_delay = 0
        else:
            self.CSI_update_delay += 1

        # print(f"距离上次更新CSI已经过去{self.CSI_update_delay}")

    def calculate_rho(self, H_old, H):
        rho_gap = np.linalg.norm(np.trace(H_old @ H.conj().T) / (np.linalg.norm(H_old, 'fro') ** 2))
        return rho_gap ** (1 / self.SRS_period)

    def generate_precoder(self):
        K = len(self.serve_UEs)
        # WMMSE
        # 先以EZF预编码初始化
        H_equal = np.zeros((self.n_stream, self.Mt), dtype=np.complex128)
        combiner = np.zeros((self.n_stream, self.Mr), dtype=np.complex128)
        for u in self.serve_UEs:
            U, s, VT = np.linalg.svd(self.H_bs_serve[u.id][0].conj().T)
            combiner[u.id * u.n_layer: (u.id + 1) * u.n_layer, :] = U[:, :u.n_layer].conj().T
            H_equal[u.id * u.n_layer: (u.id + 1) * u.n_layer, :] = combiner[u.id * u.n_layer: (u.id + 1) * u.n_layer, :] @ self.H_bs_serve[u.id][0].conj().T
        inv = np.linalg.inv(H_equal @ H_equal.conj().T + self.n_stream * self.noise / self.P * np.eye(self.n_stream))
        precoder = H_equal.conj().T @ inv
        precoder = np.sqrt(self.P / np.trace(precoder.conj().T @ precoder)) * precoder

        # # WMMSE迭代
        # combiner = np.zeros((self.Mr, self.n_stream), dtype=np.complex128)
        # for iteration in range(self.WMMSE_max_iteration):
        #     receivers = {}
        #     mse_weights = {}

        #     # 更新MMSE接收机
        #     for u in self.serve_UEs:
        #
        #         H_user = self.H_bs_serve[u.id][0]
        #
        #         W_user = precoder[:, u.id * u.n_layer: (u.id + 1) * u.n_layer]
        #
        #         # 接收信号协方差
        #         R_yy = self.noise * np.eye(self.Mr, dtype=complex)
        #         for v in self.serve_UEs:
        #             if v != u:
        #                 W_other = precoder[:, v.id * v.n_layer: (v.id + 1) * v.n_layer]
        #                 R_yy += H_user.conj().T @ W_other @ W_other.conj().T @ H_user
        #
        #         # MMSE接收机
        #         try:
        #             signal = H_user.conj().T @ W_user
        #             G_mmse = np.linalg.inv(R_yy + 1e-10 * np.eye(self.Mr)) @ signal
        #             receivers[u.id] = G_mmse
        #             combiner[:, u.id * u.n_layer: (u.id + 1) * u.n_layer] = G_mmse
        #
        #             # MSE权重
        #             I = np.eye(u.n_layer, dtype=complex)
        #             MSE_k = I - G_mmse.conj().T @ signal - signal.conj().T @ G_mmse + \
        #                     G_mmse.conj().T @ R_yy @ G_mmse
        #             mse_weights[u.id] = np.linalg.inv(MSE_k + 1e-10 * I)
        #         except:
        #             pass
        #
        #     # 更新预编码器
        #     for u in self.serve_UEs:
        #
        #         H_user = self.H_bs_serve[u.id][0]
        #
        #         # 构建干扰矩阵
        #         A = 1e-10 * np.eye(self.Mt, dtype=complex)
        #         for v in self.serve_UEs:
        #             if v != u:
        #                 H_j = self.H_bs_serve[v.id][0]
        #                 G_j = receivers[v.id]
        #                 U_j = mse_weights[v.id]
        #                 A += H_j @ G_j @ U_j @ G_j.conj().T @ H_j.conj().T
        #
        #         # 更新预编码
        #         try:
        #             B = H_user @ receivers[u.id] @ mse_weights[u.id]
        #             W_new = np.linalg.inv(A) @ B
        #
        #             # 功率约束
        #             power = np.trace(W_new @ W_new.conj().T).real
        #             if power > 0:
        #                 n_scheduled = K
        #                 power_budget = self.P / max(n_scheduled, 1)
        #                 W_new = W_new * np.sqrt(power_budget / power)
        #
        #             precoder[:, u.id * u.n_layer: (u.id + 1) * u.n_layer] = W_new
        #         except:
        #             pass
        #
        #     # 功率归一化
        #     precoder = np.sqrt(self.P / np.trace(precoder.conj().T @ precoder)) * precoder
        #
        # # combiner归一化
        # for s in range(self.n_stream):
        #     combiner[:, s] = combiner[:, s] / np.linalg.norm(combiner[:, s])
        #
        # # 我仿真时用的combiner还需要共轭转置才适配
        # combiner = combiner.conj().T

        for u in self.serve_UEs:
            self.P_user[u.id] = []
            for l in range(u.n_layer):
                self.P_user[u.id].append(np.linalg.norm(precoder[:, u.id * u.n_layer + l]) ** 2)
            u.precoder = precoder[:, u.id * u.n_layer: (u.id + 1) * u.n_layer]
            u.combiner = combiner[u.id * u.n_layer: (u.id + 1) * u.n_layer, :]

    # def choose_mcs(self, mcs_table: dict, double_precoder: bool, sinr_fix: list[float] = None):
    #     mcs_list = []
    #     postSINR_estimation_list = []
    #     if sinr_fix is None:
    #         for u in self.serve_UEs:
    #             sinr_estimate = self.postSINR_estimation(double_precoder, u)
    #             sinr_estimate += self.OLLA[u.id]
    #             postSINR_estimation_list.append(sinr_estimate)
    #             mcs = "1"
    #             for key, value in mcs_table.items():
    #                 if sinr_estimate >= value[1]:
    #                     mcs = key
    #                 else:
    #                     break
    #             mcs_list.append(mcs)
    #     else:
    #         for u in self.serve_UEs:
    #             sinr_estimate = self.postSINR_estimation(double_precoder, u)
    #             sinr_estimate += sinr_fix[u.id]
    #             postSINR_estimation_list.append(sinr_estimate)
    #             mcs = "1"
    #             for key, value in mcs_table.items():
    #                 if sinr_estimate >= value[1]:
    #                     mcs = key
    #                 else:
    #                     break
    #             mcs_list.append(mcs)
    #     return mcs_list, postSINR_estimation_list

    def choose_mcs(self, mcs_table: dict, mean_SINR_estimate: bool, OLLA_fix: list[float] = None):
        mcs_list = []
        postSINR_estimation_list = []
        postSINR_estimation_raw_list = []
        if OLLA_fix is None:
            for u in self.serve_UEs:
                sinr_estimate = self.postSINR_estimation(mean_SINR_estimate, u)
                postSINR_estimation_raw_list.append(sinr_estimate)
                sinr_estimate += self.OLLA[u.id]
                postSINR_estimation_list.append(sinr_estimate)
                mcs = "1"
                for key, value in mcs_table.items():
                    if sinr_estimate >= value[1]:
                        mcs = key
                    else:
                        break
                mcs_list.append(mcs)
        else:
            for u in self.serve_UEs:
                sinr_estimate = self.postSINR_estimation(mean_SINR_estimate, u)
                postSINR_estimation_raw_list.append(sinr_estimate)
                self.OLLA[u.id] += OLLA_fix[u.id]
                sinr_estimate += self.OLLA[u.id]
                # sinr_estimate += OLLA_fix[u.id]
                postSINR_estimation_list.append(sinr_estimate)
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
        }
        return mcs_list, info

    def postSINR_estimation(self, mean_SINR_estimate: bool, u: User):
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
                        # interference += (self.P_user[v.id][i] * v.combiner_eff_gain[i]
                        #                  * np.trace(self.Rt[v.id] @ self.Rt[u.id]).real / self.Mt)
                mu_loss = ((1 - self.rho[u.id] ** (2 * self.CSI_update_delay)) * (self.H_l_bs_serve[u.id] ** 2)
                           * interference)
                sinr_estimate_list.append(gain / (mu_loss + self.noise))
            sinr_estimate = np.exp(np.mean(np.log(np.array(sinr_estimate_list))))  # 层间几何平均
            sinr_estimate = 10 * np.log10(sinr_estimate)
        else:
            gain = (np.linalg.norm(self.H_bs_serve[u.id][0], 'fro') ** 2
                    / (self.H_bs_serve[u.id][0].shape[0] * self.H_bs_serve[u.id][0].shape[1]))
            sinr_estimate = 10 * np.log10(gain * self.P / self.noise)
            mu_loss = self.mu_loss_per_user_dB * (len(self.serve_UEs) - 1)
            layer_loss = 10 * np.log10(u.n_layer)
            sinr_estimate = sinr_estimate - mu_loss - layer_loss
        # gain = np.linalg.norm(self.H_bs_serve[u.id, :]) ** 2
        # mu_loss = np.sum(
        #     [(np.abs(self.H_bs_serve[v.id, :] @ self.H_bs_serve[u.id, :].conj().T) ** 2) / gain for v in self.serve_UEs]) - gain
        # sinr_estimate = 10 * np.log10(gain / (mu_loss + self.noise / (self.P / len(self.serve_UEs))))
        # # sinr_estimate = 10 * np.log10(gain / (mu_loss + self.noise))
        return sinr_estimate


    def get_user_idx(self, u: User):
        assert u in self.serve_UEs, "User not served!"
        return self.serve_UEs.index(u)

    def log_user(self, u: User):
        self.serve_UEs.append(u)
        self.ACK_dict[u.id] = [1 for _ in range(self.buffer_len)]
        self.OLLA[u.id] = 0

    # def update_ACK(self, u: User, ack):
    #     self.ACK_dict[u.id].append(ack)
    #     self.ACK_dict[u.id] = self.ACK_dict[u.id][1:]
    #     assert len(self.ACK_dict[u.id]) == self.buffer_len, "ACK list length is not equal to buffer_len!"
    #     if ack:
    #         self.OLLA[u.id] += self.OLLA_step * u.BLER_T
    #         if self.OLLA[u.id] >= self.OLLA_max:
    #             self.OLLA[u.id] = self.OLLA_max
    #     else:
    #         self.OLLA[u.id] -= self.OLLA_step * (1 - u.BLER_T)
    #         if self.OLLA[u.id] <= self.OLLA_min:
    #             self.OLLA[u.id] = self.OLLA_min

    def update_ACK(self, u: User, ack, OLLA_scheme):
        self.ACK_dict[u.id].append(ack)
        self.ACK_dict[u.id] = self.ACK_dict[u.id][1:]
        assert len(self.ACK_dict[u.id]) == self.buffer_len, "ACK list length is not equal to buffer_len!"
        if OLLA_scheme:
            if ack:
                self.OLLA[u.id] += self.OLLA_step * u.BLER_T
                if self.OLLA[u.id] >= self.OLLA_max:
                    self.OLLA[u.id] = self.OLLA_max
            else:
                self.OLLA[u.id] -= self.OLLA_step * (1 - u.BLER_T)
                if self.OLLA[u.id] <= self.OLLA_min:
                    self.OLLA[u.id] = self.OLLA_min
        else:
            # TODO: 为了调试纯RL方案，修改了OLLA逻辑
            if self.CSI_update_delay == 159:
                self.OLLA[u.id] = 0
            if self.OLLA[u.id] > 6:
                self.OLLA[u.id] = 6
            if self.OLLA[u.id] < - 6:
                self.OLLA[u.id] = - 6
            # if self.OLLA[u.id] >= self.OLLA_max:
            #     self.OLLA[u.id] = self.OLLA_max
            # if self.OLLA[u.id] <= self.OLLA_min:
            #     self.OLLA[u.id] = self.OLLA_min
            # else:
            #     if ack:
            #         self.OLLA[u.id] += self.OLLA_step * u.BLER_T
            #         if self.OLLA[u.id] >= self.OLLA_max:
            #             self.OLLA[u.id] = self.OLLA_max
            #     else:
            #         self.OLLA[u.id] -= self.OLLA_step * (1 - u.BLER_T)
            #         if self.OLLA[u.id] <= self.OLLA_min:
            #             self.OLLA[u.id] = self.OLLA_min


    def separate_large_scale_fading(self, H_samples):
        """
        从带有相同大尺度衰落的信道样本中分离大尺度衰落因子alpha
        :param H_samples: M×Nt×Nr 的信道样本数组，M为样本数，Nr接收天线数，Nt发射天线数
        :return: alpha: 大尺度衰落因子（复数）
                 H_small_samples: M×Nr×Nt 的小尺度衰落信道样本数组
        """
        M, Nt, Nr = H_samples.shape

        # ---------------- 分离大尺度衰落的幅度 |alpha| ----------------
        # 计算每个样本的弗罗贝尼乌斯范数平方（总功率）
        P_k = np.array([np.sum(np.abs(Hk) ** 2) for Hk in H_samples])
        # 统计平均功率
        P_avg = np.mean(P_k)
        # 计算幅度 |alpha|
        alpha = np.sqrt(P_avg / (Nr * Nt))

        # ---------------- 提取小尺度衰落信道样本 ----------------
        H_small_samples = H_samples / alpha  # 广播除法，逐样本除以alpha

        return alpha, H_small_samples

    def estimate_corr_matrix(self, H_samples):
        """
        从信道样本估计收发相关矩阵
        :param H_samples: M×Nt×Nr 信道样本数组
        :return: Rr_hat: 接收端相关矩阵 Nr×Nr
                 Rt_hat: 发射端相关矩阵 Nt×Nt
        """
        M, Nt, Nr = H_samples.shape

        # 估计接收端相关矩阵 Rr = E{H^HH}/Nt
        Rr_hat = np.zeros((Nr, Nr), dtype=np.complex_)
        for H in H_samples:
            Rr_hat += H.conj().T @ H / M
        Rr_hat /= Nt  # 归一化

        # 估计发射端相关矩阵 Rt = E{HH^H}/Nr
        Rt_hat = np.zeros((Nt, Nt), dtype=np.complex_)
        for H in H_samples:
            Rt_hat += H @ H.conj().T / M
        Rt_hat /= Nr  # 归一化

        return Rr_hat, Rt_hat

    def reset(self):
        self.Rt = {}
        self.Rr = {}
        self.serve_UEs = []
        self.ACK_dict = {}
        self.OLLA = {}
        self.H_bs_total = {}
        self.H_bs_serve = {}
        self.H_l_bs_total = {}
        self.H_l_bs_serve = {}
        self.CSI_update_delay = 0
        self.P_user = {}
        self.H_bs_real = {}
        self.n_stream = 0

class FeedbackScheduler:

    def __init__(self, D):
        self.feedback = {}
        self.D = D

    def log_user(self, u: User):
        self.feedback[u.id] = [1] * self.D

    def update(self, u: User, ack):
        self.feedback[u.id].append(ack)
        delayed_feedback = self.feedback[u.id][0]
        self.feedback[u.id] = self.feedback[u.id][1:]
        assert len(self.feedback[u.id]) == self.D, "feedback list length is not equal to feedback delay!"
        return delayed_feedback

    def reset(self, u: User):
        self.feedback[u.id] = [1] * self.D

class Environment:

    def __init__(self, cfg: Config, device):

        self._cfg = cfg
        self._device = device

        # 环境基本设置
        self._B = cfg.B              # 基站数量
        self._K = cfg.U              # 用户数量
        self._Mt = cfg.Mt              # BS天线数量
        self._Mr = cfg.Mr               # UE天线数量
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
        self._delta_theta = cfg.delta_theta/180 * np.pi     # 围绕中心角向两边扩散的最大角度
        self._kai = cfg.kai                 # 2*pi*d/λ，固定取d/λ为1/2
        # self._sigma_sh = cfg.sigma_sh     # shadow fading(dB)
        # self._delta_sh = cfg.delta_sh     # shadow fading参数

        self._noise = 10 ** (cfg.noise/10 - 3)              # 噪声功率(W/Hz)
        self._sigma = self._noise * self._Bw * 1e3 * 12          # 噪声功率(W)  TODO: 还乘了每个RB的子载波数

        self._OFDM_t = 1 / (self._Bw * 1e3)                 # OFDM符号时长
        self._slot_t = cfg.t_slot                           # 时隙时长(ms)
        # self._f_carrier = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 for n in range(self._F)]   # 各载波频率(MHz)
        # self._f_subband = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 * self._C for n in range(self._K)]   # 各子带中心频率(MHz)
        self._H = None                                      # 信道矩阵
        self._P = 10 ** (cfg.P/10 - 3)                      # 固定发射功率分配(W)

        # # 测试有干扰情况下的sinr分布情况
        # # 用户数*用户数，第一个参数表明是哪个用户，第二个参数表明存在多少用户干扰
        # self.sinr_interference_condition = [[[] for _ in range(self._K)] for _ in range(self._K)]

        # 用户设置参数
        # Users_pos = cfg.Users_pos                      # 用户位置
        self._h_U = cfg.h_U                            # 用户高度
        self._BLER_T = cfg.BLER_T                      # 用户BLER阈值
        self._N_layer = cfg.N_layer              # 用户数据流数
        # shadow_user = np.random.normal(0, 1, size=self._K)     # User侧的shadow fading参数

        # BS设置参数
        # self._max_P = cfg.max_P            # 每个BS所能承受的最大功率单元数
        # self._dP = 10 ** (cfg.max_P_dBm / 10 - 3) / cfg.max_P           # 功率分配粒度(W)
        self._h_BS = cfg.h_BS              # BS高度
        self._K_BS = cfg.K_BS              # BS最大连接用户数
        # self._N_rf = cfg.N_rf              # BS的RF chain数
        self._SRS_period = cfg.SRS_period  # SRS更新周期
        self._buffer_len = cfg.buffer_len  # BS存储历史数据条数
        self._D = cfg.feedback_delay  # ACK/NACK反馈延迟
        # shadow_BS = np.random.normal(0, 1, size=self._B)       # BS侧的shadow fading参数

        # self._Users = cfg["Users"]      # 用户数据
        # self._BSs = cfg["BSs"]          # BS数据

        self._region_bound = cfg.region_bound                # 所考虑区域大小
        self._min_distance_BS = cfg.min_distance_BS          # BS间最小间距
        self._min_distance_User = cfg.min_distance_User      # 用户间最小间距

        # self.MCS_table = {
        #     # MCS阶数: [编码效率(bits/symbol), SINR阈值(dB)]
        #     # SINR大于阈值则可以选择对应MCS，此时最优MCS为SINR恰大于本阶阈值而小于下阶阈值
        #     '1': [0.15, -6.50],
        #     '2': [0.23, -4.00],
        #     '3': [0.38, -2.60],
        #     '4': [0.60, -1.00],
        #     '5': [0.88, 1.00],
        #     '6': [1.18, 3.00],
        #     '7': [1.48, 6.60],
        #     '8': [1.91, 10.00],
        #     '9': [2.41, 11.40],
        #     '10': [2.73, 11.80],
        #     '11': [3.32, 13.00],
        #     '12': [3.90, 13.80],
        #     '13': [4.52, 15.60],
        #     '14': [5.12, 16.80],
        #     '15': [5.55, 17.60],
        # }

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

        # 是否使用SINR均值老化动态估计
        self._mean_SINR_estimate = cfg.mean_SINR_estimate
        # 是否传统OLLA机制
        self._OLLA_scheme = cfg.OLLA
        # 是否固定信道实现
        self.fix_channel = cfg.fix_channel

        if self.fix_channel:
            # quadriga信道文件地址
            self._quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/QuaDriGa/quadriga_channel_test_2"
        else:
            self._quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/QuaDriGa/quadriga_UE_packs_test"
        self._qg_src = QuadrigaChannelSource(self._quadriga_dir)

        self.feedback_scheduler = FeedbackScheduler(self._D)

        # 生成用户
        self.UEs: list[User] = []
        for i in range(self._K):
            user = User(i, self._BLER_T, self._Mr, self._N_layer)
            self.UEs.append(user)
            self.feedback_scheduler.log_user(user)

        # TODO: 指定一个ue的id，后续传输只针对单ue仿真，其余ue只作为背景板
        self.serve_ue_id = 0

        # 生成BS
        self.BSs: list[BS] = []
        for i in range(self._B):
            # temp = {"max P": self._max_P, "location": BS_pos[i], "shadow": shadow_BS[i]}
            self.BSs.append(BS(i, self._P, self._sigma, self._K_BS, self._Mt, self._Mr,
                               self._SRS_period, self._buffer_len, self._rho))

        # 记录用户MCS选择分布情况
        self.user_MCS_distribution = []
        for i in range(self._K):
            self.user_MCS_distribution.append([0] * len(self.MCS_table.keys()))

        self._slots = 0
        self._slot_bias = 0

        self._tot_bits = 0
        self.state = None

        print(f"obs dimension is {self.get_obs_dim()}")
        # 设置normalizer
        if self._cfg.obs_normalize:
            self._obs_normalizer = Normalizer((self.get_obs_dim(),), clip=25).to(self._device)
        if self._cfg.reward_normalize:
            self._reward_normalizer = Normalizer((), clip=5).to(self._device)
        if self._cfg.cost_normalize:
            self._cost_normalizer = Normalizer((self.get_cost_num(),), clip=25).to(self._device)

    def reset(self, slot_bias=0):

        self._slots = 0
        self._slot_bias = slot_bias
        self._tot_bits = 0

        # 读取quadriga信道
        self._H = {}
        for u in self.UEs:
            self._H[u.id] = self._qg_src.get_H(u.id, self._slots + self._slot_bias)  # [NRB, NBS, N_RX]
            assert (self._H[u.id].shape[1], self._H[u.id].shape[2]) == (self._Mt, self._Mr), "channel shape mismatch!"

        for bs in self.BSs:
            bs.reset()

        # 设置BS与AP关联
        # TODO: 二阶段只要求单基站，这里直接单基站处理
        for u in self.UEs:
            self.BSs[0].log_user(u)
            u.serve_BS = self.BSs[0]
            self.feedback_scheduler.reset(u)

        # TODO: 信道数据复用，若slot偏置开始从头循环，则切换到另一用户的信道来训练
        if self._slot_bias % (480 * 700) > 0:
            self.serve_ue_id = 1

        # 各个BS整理信道信息
        # 只有SRS周期时BS才会更新信道
        for bs in self.BSs:
            bs.collect_channels(self._H, self._slots)

        # 将要返回给agent的状态打包
        # ACK_list_temp = []
        # for UE_id, ack_list in self.BSs[0].ACK_dict.items():
        #     if UE_id != self.serve_ue_id:
        #         continue
        #     ACK_list_temp.append(ack_list)
        # tensor_ACK_list = torch.tensor(ACK_list_temp).flatten()
        # tensor_update_delay = torch.Tensor([self.BSs[0].CSI_update_delay])
        # tensor_OLLA = torch.Tensor([self.BSs[0].OLLA[self.serve_ue_id]])
        #
        # self.state = torch.cat((tensor_ACK_list, tensor_update_delay, tensor_OLLA), dim=0).float().to(self._device)
        u_state = []
        for u in self.UEs:
            tensor_ACK_list = torch.tensor(self.BSs[0].ACK_dict[u.id]).flatten()
            tensor_update_delay = torch.Tensor([2 * self.BSs[0].CSI_update_delay / self._SRS_period - 1])
            tensor_OLLA = torch.Tensor([self.BSs[0].OLLA[u.id]])
            u_state.append(torch.cat((tensor_ACK_list, tensor_update_delay, tensor_OLLA), dim=0).float())

        self.state = torch.stack(u_state, dim=0).to(self._device)

        info = {}

        state = self.state

        if self._cfg.obs_normalize:
            if self.fix_channel:
                origin_state = self.state
                for u in self.UEs:
                    state[u.id] = self._obs_normalizer.normalize(self.state[u.id])
            else:
                origin_state = self.state
                state[self.serve_ue_id] = self._obs_normalizer.normalize(self.state[self.serve_ue_id])
            info['origin_state'] = origin_state

        return state, info

    def step(self, action: torch.Tensor):
        """
        环境交互函数，开始时为当前slot进行发送的阶段，根据action发送完成记录数据后开启下一个slot的状态转换，再将状态交给agent进行决策

        Args:
            action: agent给出的状态，包括各个用户选择的MCS以及外层码字idx

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
            bs.generate_precoder()

        # TODO: 动作改为OLLA修正量
        OLLA_fix = self.process_action(action)
        MCS_list, info = self.BSs[0].choose_mcs(self.MCS_table, self._mean_SINR_estimate, OLLA_fix)
        postSINR_estimation_list = info["postSINR_estimation_list"]
        postSINR_estimation_raw_list = info["postSINR_estimation_raw_list"]

        # 根据调度结果更新用户数据队列，记录这一时隙内总发送比特数，并考察时延约束违反情况
        tot_bits = 0
        success_users = self._K
        ACK_list = []
        user_bits = []
        user_BLER = []
        user_BLER_ideal = []
        user_OLLA = []
        user_sinr = []
        for i, u in enumerate(self.UEs):
            bits, ACK, info = self.get_rate(u, MCS_list[i])
            ideal_bler = self.get_bler(10 ** (info["sinr"]/10), self.MCS_table[MCS_list[i]][0])
            delayed_feedback = self.feedback_scheduler.update(u, ACK)
            u.serve_BS.update_ACK(u, delayed_feedback, self._OLLA_scheme)
            u.update_BLER(ACK, self._slots, ideal_bler)

            # 记录各类参数
            if self.fix_channel:
                tot_bits += bits / self._K
            else:
                tot_bits += bits if i == self.serve_ue_id else 0
            user_bits.append(bits)
            ACK_list.append(ACK)
            user_BLER.append(u.BLER)
            user_BLER_ideal.append(u.BLER_ideal)
            user_OLLA.append(u.serve_BS.OLLA[u.id])
            user_sinr.append(info["sinr"])
            self.user_MCS_distribution[i][int(MCS_list[i])-1] += 1

        # 完美MCS选择
        ideal_MCS_list = []
        for i, u in enumerate(self.UEs):
            mcs = "1"
            for key, value in self.MCS_table.items():
                if user_sinr[i] >= value[1]:
                    mcs = key
                else:
                    break
            ideal_MCS_list.append(mcs)

        # tot_bps = tot_bits / (self._slot_t * 1e-3)
        # user_bps = [ele / (self._slot_t * 1e-3) for ele in user_bits]

        # 奖励
        # 平均估计MCS选择
        mean_MCS_list = []
        for i, u in enumerate(self.UEs):
            mcs = self.MCS_table["1"][0]
            for key, value in self.MCS_table.items():
                if postSINR_estimation_raw_list[i] >= value[1]:
                    mcs = value[0]
                else:
                    break
            mean_MCS_list.append(mcs)
        reward = tot_bits / mean_MCS_list[self.serve_ue_id] / self._N_layer
        # reward = tot_bits / self._N_layer

        # 成本
        cost_list = []
        # 直接用瞬时ACK/NACK情况结合BLER目标计算cost
        for i, u in enumerate(self.UEs):
            if i != self.serve_ue_id:
                cost_list.append(0)
            else:
                cost_u = (1 - ACK_list[i]) - u.BLER_T
                cost_list.append(cost_u)
            # cost_list.append(0)

        # 本时隙结束，下一时隙开始
        # 读取quadriga信道
        self._H = {}
        for u in self.UEs:
            self._H[u.id] = self._qg_src.get_H(u.id, self._slots + self._slot_bias)  # [NRB, NBS, N_RX]
            assert (self._H[u.id].shape[1], self._H[u.id].shape[2]) == (self._Mt, self._Mr), "channel shape mismatch!"

        # 各个BS整理信道信息
        # 只有SRS周期时BS才会更新信道
        for bs in self.BSs:
            bs.collect_channels(self._H, self._slots)

        # 将要返回给agent的状态打包
        # ACK_list_temp = []
        # for UE_id, ack_list in self.BSs[0].ACK_dict.items():
        #     if UE_id != self.serve_ue_id:
        #         continue
        #     ACK_list_temp.append(ack_list)
        # tensor_ACK_list = torch.tensor(ACK_list_temp).flatten()
        # tensor_update_delay = torch.Tensor([self.BSs[0].CSI_update_delay])
        # tensor_OLLA = torch.Tensor([self.BSs[0].OLLA[self.serve_ue_id]])
        #
        # self.state = torch.cat((tensor_ACK_list, tensor_update_delay, tensor_OLLA), dim=0).float().to(self._device)

        u_state = []
        for u in self.UEs:
            tensor_ACK_list = torch.tensor(self.BSs[0].ACK_dict[u.id]).flatten()
            tensor_update_delay = torch.Tensor([2 * self.BSs[0].CSI_update_delay / self._SRS_period - 1])
            tensor_OLLA = torch.Tensor([self.BSs[0].OLLA[u.id]])
            u_state.append(torch.cat((tensor_ACK_list, tensor_update_delay, tensor_OLLA), dim=0).float())

        self.state = torch.stack(u_state, dim=0).to(self._device)

        info = {
            'tot_bits': tot_bits,
            'cost_list': cost_list,
            'user_bits': user_bits,
            'user_BLER': user_BLER,
            'user_BLER_ideal': user_BLER_ideal,
            'user_OLLA': user_OLLA,
            'user_sinr': user_sinr,
            'user_mcs': [int(mcs) for mcs in MCS_list],
            'user_mcs_ideal': [int(mcs) for mcs in ideal_MCS_list],
            'postsinr_estimation': postSINR_estimation_list,
            'postsinr_estimation_raw': postSINR_estimation_raw_list,
            'user_MCS_distribution': self.user_MCS_distribution,
        }

        state = self.state

        if self._cfg.obs_normalize:
            if self.fix_channel:
                origin_state = self.state
                for u in self.UEs:
                    state[u.id] = self._obs_normalizer.normalize(self.state[u.id])
            else:
                origin_state = self.state
                state[self.serve_ue_id] = self._obs_normalizer.normalize(self.state[self.serve_ue_id])
            info['origin_state'] = origin_state
        else:
            info['origin_state'] = self.state

        assert not torch.isnan(state).any(), "state contains NaN values"

        reward = torch.tensor(reward).to(self._device)
        if self._cfg.reward_normalize:
            origin_reward = reward
            reward = self._reward_normalizer.normalize(reward.float())
            info['origin_reward'] = origin_reward
        else:
            info['origin_reward'] = reward

        cost = torch.tensor(cost_list[self.serve_ue_id]).to(self._device)
        if self._cfg.cost_normalize:
            origin_cost = cost
            cost = self._cost_normalizer.normalize(cost.float())
            info['origin_cost'] = origin_cost
        else:
            info['origin_cost'] = cost

        # 返回内容：输入给agent的状态、奖励、成本、性能指标
        return state, reward, cost, info

    def save(self):

        save = {}
        if self._cfg.obs_normalize:
            save['obs_normalizer'] = self._obs_normalizer
        if self._cfg.reward_normalize:
            save['reward_normalizer'] = self._reward_normalizer
        if self._cfg.cost_normalize:
            save['cost_normalizer'] = self._cost_normalizer

        return save

    def load_norm(self, **params):

        if ('obs_norm_param' in params) and self._cfg.obs_normalize:
            self._obs_normalizer.load_state_dict(params['obs_norm_param'])
        if ('reward_norm_param' in params) and self._cfg.reward_normalize:
            self._reward_normalizer.load_state_dict(params['reward_norm_param'])
        if ('cost_norm_param' in params) and self._cfg.cost_normalize:
            self._cost_normalizer.load_state_dict(params['cost_norm_param'])

    def process_action(self, action: torch.Tensor):
        # TODO: 单用户
        sinr_fix = action.tolist()
        if len(sinr_fix) == 1:
            temp = []
            for u in self.UEs:
                if u.id == self.serve_ue_id:
                    temp.append(sinr_fix[0])
                else:
                    temp.append(0)
            sinr_fix = temp
        for i, fix in enumerate(sinr_fix):
            if fix > 0.5:
                sinr_fix[i] = 0.5
            if fix < - 0.5:
                sinr_fix[i] = - 0.5
        return sinr_fix

    def get_bler(self, sinr, R):
        n = 12 * 14
        C = np.log2(1 + sinr)
        sqrt_V = np.log2(np.e) * np.sqrt((1 - 1 / ((1 + sinr) ** 2)) / n)
        return 0.5 * (1 - sp.erf(((C - R) / sqrt_V) / np.sqrt(2)))

    # 计算用户SINR
    def get_SINR(self, u: User):
        sinr_list = []
        for l in range(u.n_layer):
            up = 0
            down = 0
            combiner = u.combiner[l, :]

            for v in self.UEs:
                for i in range(v.n_layer):
                    precoding_vector = v.precoder[:, i]

                    if v == u and i == l:
                        up = np.linalg.norm(combiner @ self._H[u.id][0].conj().T @ precoding_vector) ** 2
                    else:
                        down += np.linalg.norm(combiner @ self._H[u.id][0].conj().T @ precoding_vector) ** 2

            sinr_list.append(up / (down + self._sigma))
        sinr = np.exp(np.mean(np.log(np.array(sinr_list))))  # 层间几何平均

        info = {
            "gain": up,
            "interference": down,
            "noise + interference": down + self._sigma,
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

        sinr = 10 * np.log10(sinr)
        if sinr >= self.MCS_table[mcs][1]:
            ACK = 1
            bits = self.MCS_table[mcs][0] * self._N_layer
        else:
            ACK = 0
            bits = 0
        info["sinr"] = sinr

        return bits, ACK, info

    # 得到环境参数
    def get_obs_dim(self):
        # TODO: 单用户
        return self._buffer_len + 1 + 1

    def get_act_dim(self):
        # TODO: 单用户
        return 1

    def get_cost_num(self):
        return 1
