from scipy.io import loadmat
import numpy as np
import os
from functools import lru_cache

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


# def estimate_rx_corr_per_tx(H_samples):
#     """
#     估计每个发射天线对应的接收相关矩阵 R_{r,j}
#     :param H_samples: M×Nt×Nr 信道样本数组
#     :return: Rr_list: 长度为Nt的列表，每个元素是Nr×Nr的 R_{r,j}
#              Rr_avg: Nr×Nr 平均接收相关矩阵
#     """
#     M, Nt, Nr = H_samples.shape
#     Rr_list = []
#
#     # 遍历每个发射天线，计算对应的接收相关矩阵
#     for j in range(Nt):
#         # 提取第j个发射天线的所有接收列向量（M×Nr）
#         hj_samples = H_samples[:, j, :]
#         # 计算相关矩阵 R_{r,j} = E{h_j h_j^H}
#         Rrj = (hj_samples.conj().T @ hj_samples) / M
#         # 归一化（确保对角元为1，消除功率差异影响）
#         diag_Rrj = np.diag(Rrj).real
#         diag_inv_sqrt = np.diag(1 / np.sqrt(diag_Rrj))  # 避免除零
#         Rrj_norm = diag_inv_sqrt @ Rrj @ diag_inv_sqrt
#         Rr_list.append(Rrj_norm)
#
#     # 计算平均接收相关矩阵
#     Rr_avg = np.mean(Rr_list, axis=0)
#
#     return Rr_list, Rr_avg
#
#
# def calculate_rrj_mse(Rr_list):
#     """
#     计算所有 R_{r,j} 之间的两两MSE，以及相对MSE
#     :param Rr_list: 长度为Nt的接收相关矩阵列表
#     :return: pairwise_mse: 所有两两组合的MSE列表
#              avg_mse: 平均MSE（核心验证指标）
#              rel_mse: 相对MSE（avg_mse / 平均迹，消除尺度影响）
#     """
#     Nt = len(Rr_list)
#     Nr = Rr_list[0].shape[0]
#     pairwise_mse = []
#
#     # 计算所有两两组合的MSE（i<j，避免重复）
#     for i in range(Nt):
#         for j in range(i + 1, Nt):
#             Rri = Rr_list[i]
#             Rrj = Rr_list[j]
#             # 逐元素MSE
#             mse_ij = np.sum(np.abs(Rri - Rrj) ** 2) / (Nr * Nr)
#             pairwise_mse.append(mse_ij)
#
#     # 计算平均MSE和相对MSE
#     avg_mse = np.mean(pairwise_mse)
#     # 计算平均迹（归一化因子）
#     avg_trace = np.mean([np.trace(np.abs(Rrj)) / Nr for Rrj in Rr_list])
#     rel_mse = avg_mse / avg_trace
#
#     return pairwise_mse, avg_mse, rel_mse

def estimate_kronecker_corr(H_samples):
    """
    从信道样本估计Kronecker模型的收发相关矩阵
    :param H_samples: M×Nt×Nr 信道样本数组
    :return: R_hat: 整体协方差矩阵 (Nr*Nt)×(Nr*Nt)
             Rr_hat: 接收端相关矩阵 Nr×Nr
             Rt_hat: 发射端相关矩阵 Nt×Nt
    """
    M, Nt, Nr = H_samples.shape

    # 步骤1：估计整体协方差矩阵 R = E{vec(H)vec(H)^H}
    vec_H = []
    for H in H_samples:
        vec_h = np.reshape(H, (Nr * Nt, 1), order='F')  # vec(H)，列优先
        vec_H.append(vec_h)
    vec_H = np.hstack(vec_H)  # (Nr*Nt)×M
    R_hat = (vec_H @ vec_H.conj().T) / M  # (Nr*Nt)×(Nr*Nt)

    # 步骤2：估计接收端相关矩阵 Rr = E{H^HH}/Nt
    Rr_hat = np.zeros((Nr, Nr), dtype=np.complex_)
    for H in H_samples:
        Rr_hat += H.conj().T @ H / M
    Rr_hat /= Nt  # 归一化

    # 步骤3：估计发射端相关矩阵 Rt = E{HH^H}/Nr
    Rt_hat = np.zeros((Nt, Nt), dtype=np.complex_)
    for H in H_samples:
        Rt_hat += H @ H.conj().T / M
    Rt_hat /= Nr  # 归一化

    return R_hat, Rr_hat, Rt_hat


def validate_kronecker_assumption(R_hat, Rr_hat, Rt_hat):
    """
    验证Kronecker假设：计算拟合误差+一致性指标
    :param R_hat: 真实整体协方差矩阵 (Nr*Nt)×(Nr*Nt)
    :param Rr_hat: 估计的接收相关矩阵 Nr×Nr
    :param Rt_hat: 估计的发射相关矩阵 Nt×Nt
    :return: mse: 均方误差 (R_hat - Rr⊗Rt) 的归一化MSE
             nmse: 归一化均方误差 (MSE / 迹(R_hat))
             corr_coeff: 相关系数（越接近1，Kronecker假设越成立）
    """
    # 步骤1：计算 Kronecker 乘积 Rr⊗Rt
    R_kronecker = np.kron(Rr_hat, Rt_hat)

    # 步骤2：计算均方误差 MSE
    err = R_hat - R_kronecker
    mse = np.sum(np.abs(err) ** 2) / err.size  # 平均MSE

    # 步骤3：归一化MSE（消除功率影响）
    trace_R = np.trace(R_hat) / R_hat.shape[0]  # 平均迹
    nmse = mse / trace_R

    # 步骤4：计算相关系数（衡量拟合程度）
    vec_R = np.reshape(R_hat, (-1, 1), order='F')
    vec_Rk = np.reshape(R_kronecker, (-1, 1), order='F')
    corr_numerator = np.abs(vec_R.conj().T @ vec_Rk)[0, 0]
    corr_denominator = np.sqrt((vec_R.conj().T @ vec_R)[0, 0] * (vec_Rk.conj().T @ vec_Rk)[0, 0])
    corr_coeff = corr_numerator / corr_denominator

    return mse, nmse, corr_coeff


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


K = 2
Mt = 256
Mr = 4
N_RB = 96
n_layer = 1     # 每个用户传输流数
n_stream = K * n_layer
WMMSE_max_iteration = 5     # WMMSE最大迭代计算次数
kai = np.pi
delta_theta = 5/180 * np.pi
BS_pos = [300, 300]
u_pos = [[[ 52, 201 ], [ 178, 262 ]], [[ 52, 201 ], [ 178, 62 ]], [[ 52, 201 ], [ 178, 562 ]]]
h_U = 1.65
h_BS = 15
fc = 3500
noise = 10 ** (-142/10 - 3) * 30 * 1e3
P0 = 10 ** (10/10 - 3)

T = 160
N = 50
I = 0.013
rho = 0.9966

quadriga_dir = "../信道/QuaDriGa/quadriga_channel_test"

# 读取quadriga信道
qg_src = QuadrigaChannelSource(quadriga_dir)
H = {}
Hl = {}
for ue_id in range(K):
    H[ue_id] = qg_src.get_H(ue_id, 0)       # [NRB, Mt, Mr]
    assert H[ue_id].shape == (N_RB, Mt, Mr), "channel shape mismatch!"

    Hl[ue_id], Hs = separate_large_scale_fading(H[ue_id])

    # # 验证不同天线的接收相关矩阵之间的独立性
    # Rr_list, Rr_avg = estimate_rx_corr_per_tx(H[ue_id])
    # pairwise_mse, avg_mse, rel_mse = calculate_rrj_mse(Rr_list)
    # print("="*30 + f"user {ue_id}" + "="*30)
    # print(f"  所有R_{{r,j}}的平均两两MSE：{avg_mse:.4e}")
    # print(f"  相对MSE（归一化）：{rel_mse:.4e}")
    # R_hat, Rr_hat, Rt_hat = estimate_kronecker_corr(Hs)
    # mse, nmse, corr = validate_kronecker_assumption(R_hat, Rr_hat, Rt_hat)
    # print("=" * 30 + f"user {ue_id}" + "=" * 30)
    # print(f"  均方误差 (MSE)：{mse:.4e}")
    # print(f"  归一化MSE：{nmse:.4e}")
    # print(f"  拟合相关系数：{corr:.4f}")

# def generate_gaussian_channel(Rr, Rt, n_layers):
#
#     Mt = Rt.shape[1]
#     Mr = Rr.shape[1]
#     K = Rt.shape[0]
#     h = np.zeros((K, Mr, Mt), dtype=np.complex128)
#     combiner_statistical = np.zeros((K, n_layers, Mr), dtype=np.complex128)
#     combiner_equal_gain = np.zeros((K, n_layers), dtype=np.complex128)
#
#     for k in range(K):
#         # 步骤1：特征值分解（对 Hermitian 矩阵用 eigh，确保数值稳定）
#         eigenvalues_t, eigenvectors_t = np.linalg.eigh(Rt[k])
#
#         # 步骤2：处理特征值（确保非负，避免数值误差导致的微小负数）
#         eigenvalues_t = eigenvalues_t[::-1]
#         eigenvectors_t = eigenvectors_t[:, ::-1]
#         epsilon = 1e-18
#         eigenvalues_t = np.maximum(eigenvalues_t, epsilon)  # 替换负特征值为极小正数
#         lambda_sqrt = np.sqrt(eigenvalues_t)  # 特征值开平方
#
#         # 步骤3：构造平方根矩阵 U * Lambda^(1/2)
#         Rt_square = eigenvectors_t @ np.diag(lambda_sqrt)
#
#         eigenvalues_r, eigenvectors_r = np.linalg.eigh(Rr[k])
#
#         eigenvalues_r = eigenvalues_r[::-1]
#         eigenvectors_r = eigenvectors_r[:, ::-1]
#         epsilon = 1e-18
#         eigenvalues_r = np.maximum(eigenvalues_r, epsilon)
#         lambda_sqrt = np.sqrt(eigenvalues_r)
#
#         Rr_square = eigenvectors_r @ np.diag(lambda_sqrt)
#         combiner_statistical[k] = eigenvectors_r[:, :n_layers].conj().T
#         combiner_equal_gain[k] = lambda_sqrt[:n_layers]
#
#         # 一次性生成所有随机数
#         h_real = np.random.randn(1, Mr, Mt)
#         h_imag = np.random.randn(1, Mr, Mt)
#         h0 = (h_real + 1j * h_imag) / np.sqrt(2)
#
#         # 批量变换
#         h[k, :, :] = Rr_square @ h0 @ Rt_square
#
#     return h, combiner_statistical, combiner_equal_gain
#
# def calculate_AOD(x1, y1, x2, y2):
#     dx, dy = x2 - x1, y2 - y1
#     theta_bar = np.arctan2(dx, dy)
#     return np.abs(theta_bar)
#
# pos_idx = 0
# # 生成信道空间相关矩阵
# Rt = np.zeros((K, Mt, Mt), dtype=np.complex128)
# for k in range(K):
#     for m in range(Mt):
#         for n in range(Mt):
#             theta_bar = calculate_AOD(BS_pos[0], BS_pos[1], u_pos[pos_idx][k][0], u_pos[pos_idx][k][1])
#             Rt[k][m][n] = 1 / Mt * np.exp(
#                 -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                 kai * (m - n) * delta_theta * np.sin(theta_bar))
#
# Rr = np.zeros((K, Mr, Mr), dtype=np.complex128)
# for k in range(K):
#     for m in range(Mr):
#         for n in range(Mr):
#             theta_bar = calculate_AOD(u_pos[pos_idx][k][0], u_pos[pos_idx][k][1], BS_pos[0], BS_pos[1])
#             Rr[k][m][n] = 1 / Mr * np.exp(
#                 -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                 kai * (m - n) * delta_theta * np.sin(theta_bar))
#
# H = np.zeros((K, N, Mr, Mt), dtype=np.complex128)
# for n in range(N):
#     Hs, _, combiner_equal_gain = generate_gaussian_channel(Rr, Rt, n_layer)
#     H[:, n, :, :] = Hs
#
# for ue_id in range(K):
#     # 验证不同天线的接收相关矩阵之间的独立性
#     Rr_list, Rr_avg = estimate_rx_corr_per_tx(H[ue_id, :, :, :])
#     pairwise_mse, avg_mse, rel_mse = calculate_rrj_mse(Rr_list)
#     print("="*30 + f"user {ue_id}" + "="*30)
#     print(f"  所有R_{{r,j}}的平均两两MSE：{avg_mse:.4e}")
#     print(f"  相对MSE（归一化）：{rel_mse:.4e}")