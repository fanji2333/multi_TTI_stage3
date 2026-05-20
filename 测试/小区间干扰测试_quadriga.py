import os
from functools import lru_cache
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def _scalar(x):
    import numpy as _np
    return int(_np.array(x).squeeze())

@lru_cache(maxsize=1)
def _load_meta(out_dir: str):
    m = loadmat(os.path.join(out_dir, "meta.mat"))
    meta = {}
    # 新增了 "B", "U_total", 移除了原来的 "U"
    for k in ["PACK_SIZE", "NRB", "NBS", "N_RX", "Ns_per_UE", "U_total", "B", "fc", "BW", "SCS"]:
        if k in m:
            meta[k] = _scalar(m[k])
    return meta

def _pack_path(out_dir: str, bs_id_1based: int, ue_id_1based: int, pack_id: int) -> str:
    # 匹配 MATLAB 中 BSxx_UExx_packxxx.mat 的命名规则
    return os.path.join(out_dir, f"BS{bs_id_1based:02d}_UE{ue_id_1based:02d}_pack{pack_id:03d}.mat")

@lru_cache(maxsize=16)
def _load_pack(out_dir: str, bs_id_1based: int, ue_id_1based: int, pack_id: int):
    fn = _pack_path(out_dir, bs_id_1based, ue_id_1based, pack_id)
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"not found: {fn}")

    d = loadmat(fn, variable_names=["H_batch"])
    Hb = d["H_batch"]  # [K, NRB, NBS, N_RX] complex single
    import numpy as _np
    if Hb.dtype != _np.complex64:
        Hb = Hb.astype(_np.complex64, copy=False)
    return Hb


class QuadrigaMultiCellChannelSource:
    """按需读取：给定 bs_id(0-based), ue_id(0-based) 和 tti(0-based)，返回 H"""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.meta = _load_meta(out_dir)
        self.NRB = self.meta["NRB"]
        self.NBS = self.meta["NBS"]
        self.NRX = self.meta["N_RX"]
        self.PS = self.meta["PACK_SIZE"]
        self.Ns = self.meta["Ns_per_UE"]
        self.B = self.meta.get("B", 1)  # 基站总数
        self.U_total = self.meta.get("U_total", 1)  # 场景内所有小区的总用户数

    def get_H(self, bs_id_0based: int, ue_id_0based: int, tti_0based: int):
        """获取特定 BS 到特定 UE 的信道 [NRB, NBS, N_RX]"""
        bs1 = bs_id_0based + 1
        ue1 = ue_id_0based + 1
        tti1 = (tti_0based % self.Ns) + 1
        pack_id = (tti1 - 1) // self.PS + 1
        idx_in_pack = (tti1 - 1) % self.PS

        Hb = _load_pack(self.out_dir, bs1, ue1, pack_id)  # [K, NRB, NBS, N_RX]
        if idx_in_pack >= Hb.shape[0]:
            raise IndexError(f"TTI {tti1} not present in pack #{pack_id} (size {Hb.shape[0]})")

        H = Hb[idx_in_pack, :, :, :]  # [NRB, NBS, N_RX]
        return H

    def get_H_all_BS(self, ue_id_0based: int, tti_0based: int):
        """
        [进阶功能] 一次性提取所有基站到目标 UE 的信道。
        返回形如 [B, NRB, NBS, N_RX] 的 NumPy 数组。
        非常适合用于后续直接切片：
        H_serve = H_all[serving_bs_id]
        H_interf = H_all[interfering_bs_ids]
        """
        import numpy as _np
        H_all = _np.zeros((self.B, self.NRB, self.NBS, self.NRX), dtype=_np.complex64)
        for b in range(self.B):
            H_all[b] = self.get_H(bs_id_0based=b, ue_id_0based=ue_id_0based, tti_0based=tti_0based)
        return H_all

def separate_large_scale_fading(H_samples):
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

def estimate_corr_matrix(H_samples):
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


Mt, Mr = 256, 4
h_U, h_BS = 1.65, 15
region_bound = 300
fc = 6700   # MHz
K = 2   # 单个cell内用户数
B = 3   # 总cell数
BS_pos = [(0, 0), (300, 0), ()]
n_layer = 2
n_stream = n_layer * K
noise0 = 10 ** (-174/10 - 3)              # 噪声功率(W/Hz)
noise = noise0 * 30 * 1e3 * 12
P = 10 ** (23.18/10 - 3)
WMMSE_max_iteration = 5

# quadriga信道文件地址
quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/QuaDriGa/quadriga_multicell_channel_out_2"
qg_src = QuadrigaMultiCellChannelSource(quadriga_dir)

pack_len = 1600
start_pack = pack_len * 0
packs_num = 10 - start_pack // pack_len
# 统计ICI
ICI = {}
ICI_mean = {}
ICI_mean_estimate = {}

BS_serve = {}
for b in range(B):
    BS_serve[b] = set()
    for k in range(K * B):
        if k // 2 == b:
            BS_serve[b].add(k)

for sample in range(pack_len):
    print(f"now running sample: {sample}")

    # 读取quadriga信道
    H = {}
    Hl = {}
    Hs = {}
    Rt = {}
    Rr = {}

    for b in range(B):
        for k in range(K * B):
            slots = start_pack + sample
            H[(b, k)] = qg_src.get_H(b, k, slots)      # [NRB, NBS, N_RX]
            assert (H[(b, k)].shape[1], H[(b, k)].shape[2]) == (Mt, Mr), "channel shape mismatch!"
            # 得到小尺度衰落
            Hl[(b, k)], Hs[(b, k)] = separate_large_scale_fading(H[(b, k)])
            if sample == 0:
                Rr[(b, k)], Rt[(b, k)] = estimate_corr_matrix(Hs[(b, k)])

    # 只需要在信道更新时计算预编码
    if sample == 0:
        # 各个基站做预编码
        precoder_all = {}
        combiner_all = {}
        v_all = {}
        for b in range(B):
            # WMMSE
            # 先以EZF预编码初始化
            H_equal = np.zeros((n_stream, Mt), dtype=np.complex128)
            combiner = np.zeros((n_stream, Mr), dtype=np.complex128)
            for k in range(K * B):
                if k in BS_serve[b]:
                    U, s, VT = np.linalg.svd(H[(b, k)][0].conj().T)
                    combiner[k % 2 * n_layer: (k % 2 + 1) * n_layer, :] = U[:, :n_layer].conj().T
                    H_equal[k % 2 * n_layer: (k % 2 + 1) * n_layer, :] = combiner[k % 2 * n_layer: (k % 2 + 1) * n_layer, :] @ H[(b, k)][0].conj().T
            inv = np.linalg.inv(H_equal @ H_equal.conj().T + n_stream * noise / P * np.eye(n_stream))
            precoder = H_equal.conj().T @ inv
            precoder = np.sqrt(P / np.trace(precoder.conj().T @ precoder)) * precoder

            # WMMSE迭代
            combiner = np.zeros((Mr, n_stream), dtype=np.complex128)
            for iteration in range(WMMSE_max_iteration):
                receivers = {}
                mse_weights = {}

                # 更新MMSE接收机
                for k in range(K * B):
                    if k not in BS_serve[b]:
                        continue

                    H_user = H[(b, k)][0]

                    W_user = precoder[:, k % 2 * n_layer: (k % 2 + 1) * n_layer]

                    # 接收信号协方差
                    R_yy = noise * np.eye(Mr, dtype=complex)
                    for v in BS_serve[b]:
                        if v != k:
                            W_other = precoder[:, v % 2 * n_layer: (v % 2 + 1) * n_layer]
                            R_yy += H_user.conj().T @ W_other @ W_other.conj().T @ H_user

                    # MMSE接收机
                    try:
                        signal = H_user.conj().T @ W_user
                        G_mmse = np.linalg.inv(R_yy + 1e-10 * np.eye(Mr)) @ signal
                        receivers[k % 2] = G_mmse
                        combiner[:, k % 2 * n_layer: (k % 2 + 1) * n_layer] = G_mmse

                        # MSE权重
                        I = np.eye(n_layer, dtype=complex)
                        MSE_k = I - G_mmse.conj().T @ signal - signal.conj().T @ G_mmse + \
                                G_mmse.conj().T @ R_yy @ G_mmse
                        mse_weights[k % 2] = np.linalg.inv(MSE_k + 1e-10 * I)
                    except:
                        pass

                # 更新预编码器
                for k in range(K * B):
                    if k not in BS_serve[b]:
                        continue

                    H_user = H[(b, k)][0]

                    # 构建干扰矩阵
                    A_m = 1e-10 * np.eye(Mt, dtype=complex)
                    for v in BS_serve[b]:
                        if v != k:
                            H_j = H[(b, v)][0]
                            G_j = receivers[v % 2]
                            U_j = mse_weights[v % 2]
                            A_m += H_j @ G_j @ U_j @ G_j.conj().T @ H_j.conj().T

                    # 更新预编码
                    try:
                        B_m = H_user @ receivers[k % 2] @ mse_weights[k % 2]
                        W_new = np.linalg.inv(A_m) @ B_m

                        # 功率约束
                        power = np.trace(W_new @ W_new.conj().T).real
                        if power > 0:
                            n_scheduled = K
                            power_budget = P / max(n_scheduled, 1)
                            W_new = W_new * np.sqrt(power_budget / power)

                        precoder[:, k % 2 * n_layer: (k % 2 + 1) * n_layer] = W_new
                    except:
                        pass

            # 功率归一化
            precoder = np.sqrt(P / np.trace(precoder.conj().T @ precoder)) * precoder

            # combiner归一化
            for s in range(n_stream):
                combiner[:, s] = combiner[:, s] / np.linalg.norm(combiner[:, s])

            # 我仿真时用的combiner还需要共轭转置才适配
            combiner = combiner.conj().T

            for k in range(K * B):
                if k not in BS_serve[b]:
                    continue
                precoder_all[k] = precoder[:, k % 2 * n_layer: (k % 2 + 1) * n_layer]
                combiner_all[k] = combiner[k % 2 * n_layer: (k % 2 + 1) * n_layer, :]
                for l in range(n_layer):
                    _, eigenvectors = np.linalg.eigh(Rr[(b, k)])
                    v_all[(k, l)] = eigenvectors[:, -(l+1)]

    for k in range(K * B):
        for b in range(B):
            if k in BS_serve[b]:
                continue

            for l in range(n_layer):
                combiner = combiner_all[k][l, :]
                for v in BS_serve[b]:
                    for i in range(n_layer):
                        precoder = precoder_all[v][:, i]
                        # precoder归一化
                        precoder /= np.linalg.norm(precoder)
                        tmp = ICI.get((k, l, b, v), [])
                        tmp.append(np.linalg.norm(combiner @ Hs[(b, k)][0].conj().T @ precoder) ** 2)
                        ICI[(k, l, b, v)] = tmp
                    if sample == 0:
                        ICI_mean_estimate[(k, l, b, v)] = 1 / Mt * ((v_all[(k, l)].conj().T @ Rr[(b, k)] @ v_all[(k, l)]) * np.trace(Rt[(b, v)] @ Rt[(b, k)])).real
                        ICI_mean[(k, l, b, v)] = 0
                        for i in range(n_layer):
                            precoder = precoder_all[v][:, i]
                            precoder /= np.linalg.norm(precoder)
                            ICI_mean[(k, l, b, v)] += (combiner @ Rr[(b, k)] @ combiner.conj().T).real * (precoder.conj().T @ Rt[(b, k)] @ precoder).real
                        ICI_mean[(k, l, b, v)] /= n_layer

num_bins = 100  # 直方图的柱子数量
r = 5
bin_range = (0, r)  # 数据范围
plt.ylim(0, r)

for k in range(K * B):
    for b in range(B):
        if k in BS_serve[b]:
            continue

        for l in range(n_layer):
            for v in BS_serve[b]:
                # 绘制直方图
                plt.figure()
                n, bins, patches = plt.hist(ICI[(k, l, b, v)], num_bins, bin_range, density=True, alpha=0.6, color='g')
                lambd = 1 / ICI_mean_estimate[(k, l, b, v)]
                x = np.linspace(0, r, 200*r)
                y = np.where(x >= 0, lambd * np.exp(-lambd * x), 0)
                plt.plot(x, y, label=f'mean {ICI_mean_estimate[(k, l, b, v)]:.3f} (estimated)', color='r')
                lambd = 1 / ICI_mean[(k, l, b, v)]
                y = np.where(x >= 0, lambd * np.exp(-lambd * x), 0)
                plt.plot(x, y, label=f'mean {ICI_mean[(k, l, b, v)]:.3f} (analytical)', color='brown')
                y = np.where(x >= 0, np.exp(-x), 0)
                plt.plot(x, y, label=f'mean {1}', color='b')
                plt.xlabel('ICI')
                plt.ylabel('Frequency')
                plt.legend()
                plt.title(f'ICI Histogram for {l}-th stream of UE {k} from BS {b} UE {v}')
                plt.show()

                # 绘制时间图
                plt.figure()
                plt.plot(ICI[(k, l, b, v)], label='ICI')
                plt.axhline(y=ICI_mean_estimate[(k, l, b, v)], color='red', linestyle='--', linewidth=1, label='mean estimate')
                plt.legend()
                plt.xlabel('TTI')
                plt.ylabel('ICI')
                plt.title(f'ICI of {l}-th stream of UE {k} from BS {b} UE {v}')
                plt.show()