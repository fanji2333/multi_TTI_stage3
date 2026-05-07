from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
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
noise = 10 ** (-174/10 - 3) * 30 * 1e3 * 12
P0 = 10 ** (10/10 - 3)
T = 160
start_TTI = 160
quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/quadriga_UE_packs_test"

# 读取quadriga信道
qg_src = QuadrigaChannelSource(quadriga_dir)
rho = {}
rho_exp = {}
hl = {}
mrt_gain = {}
ezf_gain = {}
combiner = {}
precoder = {}
#
# H_equal = np.zeros((K, Mt), dtype=np.complex128)
# for ue_id in range(K):
#     H = qg_src.get_H(ue_id, start_TTI)  # [NRB, NBS, N_RX]
#     assert H.shape == (N_RB, Mt, Mr), "channel shape mismatch!"
#     # 提取小尺度衰落
#     Hl, Hs = separate_large_scale_fading(H)
#
#     U, s, VT = np.linalg.svd(H[0].conj().T)
#     combiner[ue_id] = U[:, 0].conj().T
#     H_equal[ue_id, :] = combiner[ue_id] @ H[0].conj().T
# inv = np.linalg.inv(H_equal @ H_equal.conj().T + K * noise / P0 * np.eye(K))
# precoder_temp = H_equal.conj().T @ inv
# precoder_temp = np.sqrt(P0 / np.trace(precoder_temp.conj().T @ precoder_temp)) * precoder_temp
# for ue_id in range(K):
#     precoder[ue_id] = precoder_temp[:, ue_id]

for ue_id in range(K):
    rho[ue_id] = []
    hl[ue_id] = []
    mrt_gain[ue_id]= []
    ezf_gain[ue_id] = []
    rho_exp[ue_id] = [0.99746]
    for t in range(T):
        H = qg_src.get_H(ue_id, t + start_TTI)       # [NRB, NBS, N_RX]
        # assert H.shape == (N_RB, Mt, Mr), "channel shape mismatch!"
        # 提取小尺度衰落
        Hl, Hs = separate_large_scale_fading(H)

        hl[ue_id].append(Hl)

        Hs_temp = Hs[0][:, 0]

        if t == 0:
            prev_Hs = Hs[0]
            # mrt_precoder = Hs_temp
            # U, s, VT = np.linalg.svd(Hs[0].conj().T)
            # combiner = U[:, 0].conj().T
            # H_equal = combiner @ Hs[0].conj().T
            # precoder_temp = H_equal.conj().T
            # ezf_gain[ue_id].append(np.linalg.norm(combiner[ue_id] @ H[0].conj().T @ precoder[ue_id]) ** 2)
            # ezf_gain[ue_id].append(combiner[ue_id] @ H[0].conj().T @ precoder[ue_id])
            continue
        else:
            # rho_ij = np.zeros((Mt, Mr), dtype=np.complex64)
            # for i in range(Mt):
            #     for j in range(Mr):
            #         # temp1 = prev_Hs[i, j] * Hs[0][i, j].conj()
            #         # temp2 = (np.abs(prev_Hs[i, j]) * np.abs(Hs[0][i, j]))
            #         # temp3 = temp1 / temp2
            #         rho_ij[i, j] = prev_Hs[i, j] * Hs[0][i, j].conj() / (np.abs(prev_Hs[i, j]) * np.abs(Hs[0][i, j]))
            # prev_Hs = Hs[0]
            # rho_ij = np.mean(rho_ij)
            rho_ij = np.trace(prev_Hs @ Hs[0].conj().T) / np.linalg.norm(prev_Hs, 'fro') ** 2
            rho[ue_id].append(10 * np.log10(np.abs(rho_ij) ** 2))
            # rho[ue_id].append(np.abs(rho_ij))
            rho_exp[ue_id].append(rho_exp[ue_id][0] ** t)

            # mrt_gain[ue_id].append(np.linalg.norm(Hs_temp.conjugate().T @ mrt_precoder) ** 2)
            # ezf_gain[ue_id].append(np.linalg.norm(combiner[ue_id] @ H[0].conj().T @ precoder[ue_id]) ** 2)
            # ezf_gain[ue_id].append(combiner[ue_id] @ H[0].conj().T @ precoder[ue_id])

# plt.plot(10 * np.log10(np.array([r ** 2 * 0.6 + 0.4  for r in rho[0]])), label="UE 0")
# plt.plot(10 * np.log10(np.array([r ** 2 * 0.8 + 0.2  for r in rho[1]])), label="UE 1")
plt.plot(rho[0], label="UE 0")
plt.plot(rho[1], label="UE 1")
plt.plot(rho_exp[0], label="UE 0 exp estimation")
plt.plot(rho_exp[1], label="UE 1 exp estimation")
plt.legend()
plt.grid(True)
plt.title(f"norm correlation (start TTI={start_TTI})")
# plt.ylim(0.995, 1)
plt.show()

# plt.figure()
# plt.plot(hl[0], label="UE 0")
# plt.plot(hl[1], label="UE 1")
# plt.legend()
# plt.grid(True)
# plt.show()
print("done")

# plt.figure()
# plt.plot(mrt_gain[0], label="UE 0")
# plt.plot(mrt_gain[1], label="UE 1")
# plt.legend()
# plt.grid(True)
# plt.title(f"RX antenna 0 MRT gain (start TTI={start_TTI})")
# # plt.ylim(-30, 0)
# plt.show()