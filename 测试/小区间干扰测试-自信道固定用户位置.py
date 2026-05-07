import numpy as np
import matplotlib.pyplot as plt
from common.tools import get_dis

Mt, Mr = 256, 4
h_U, h_BS = 1.65, 15
region_bound = 300
fc = 6700   # MHz
K = 2   # 单个cell内用户数
B = 2   # 总cell数
BS_pos = [(0, 0), (300, 0)]
n_layer = 2
n_stream = n_layer * K

delta_theta = 5 / 180 * np.pi
kai = 3.141592653589793

noise0 = 10 ** (-174/10 - 3)              # 噪声功率(W/Hz)
noise = noise0 * 30 * 1e3 * 12
P = 10 ** (23.18/10 - 3)
WMMSE_max_iteration = 5

def get_largescale(dub, fc):
    beta_dB = 31.84 + 21.5 * np.log10(dub) + 19 * np.log10(fc * 1e-3)
    return 10 ** (- beta_dB / 10)

def calculate_AOD(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    theta_bar = np.arctan2(dx, dy)
    return np.abs(theta_bar)

def generate_gaussian_channel(Rr, Rt, n_layers):

    Mt = Rt.shape[0]
    Mr = Rr.shape[0]

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

    # 类似方式构造接收相关矩阵平方根
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

# 统计ICI
ICI = {}
ICI_mean = {}
ICI_mean_estimate = {}
samples = 10000   #随机实现次数

# 随机初始化用户位置
User_pos = []
BS_serve = {}
for b in range(B):
    BS_serve[b] = set()
    for k in range(K * B):
        if k // 2 == b:
            User_pos.append((BS_pos[b][0] + (np.random.rand() - 0.5) * region_bound, BS_pos[b][1] + (np.random.rand() - 0.5) * region_bound))
            BS_serve[b].add(k)

# 生成大尺度衰落
Hl = {}
for b in range(B):
    for k in range(K * B):
        dub = get_dis(User_pos[k], BS_pos[b], h_U, h_BS)
        Hl[(b, k)] = np.sqrt(get_largescale(dub, fc))

# 生成信道空间相关矩阵
Rt = {}
for b in range(B):
    Rt[b] = {}
    for k in range(K * B):
        Rt[b][k] = np.zeros((Mt, Mt), dtype=np.complex128)
        for m in range(Mt):
            for n in range(Mt):
                theta_bar = calculate_AOD(BS_pos[b][0], BS_pos[b][1], User_pos[k][0], User_pos[k][1])
                Rt[b][k][m][n] = 1 / Mt * np.exp(
                    -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
                    kai * (m - n) * delta_theta * np.sin(theta_bar))

Rr = {}
for b in range(B):
    Rr[b] = {}
    for k in range(K * B):
        Rr[b][k] = np.zeros((Mr, Mr), dtype=np.complex128)
        for m in range(Mr):
            for n in range(Mr):
                theta_bar = calculate_AOD(BS_pos[b][0], BS_pos[b][1], User_pos[k][0], User_pos[k][1])
                Rr[b][k][m][n] = 1 / Mr * np.exp(
                    -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
                    kai * (m - n) * delta_theta * np.sin(theta_bar))

for sample in range(samples):
    print(f"now running sample: {sample}")

    # 生成大尺度与小尺度衰落
    H = {}
    Hs = {}
    for b in range(B):
        for k in range(K * B):
            Hs[(b, k)], _, _ = generate_gaussian_channel(Rr[b][k], Rt[b][k], n_layer)
            H[(b, k)] = Hl[(b, k)] * Hs[(b, k)]

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
                    U, s, VT = np.linalg.svd(H[(b, k)].conj().T)
                    combiner[k % 2 * n_layer: (k % 2 + 1) * n_layer, :] = U[:, :n_layer].conj().T
                    H_equal[k % 2 * n_layer: (k % 2 + 1) * n_layer, :] = combiner[k % 2 * n_layer: (k % 2 + 1) * n_layer, :] @ \
                                                                           H[(b, k)].conj().T
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

                    H_user = H[(b, k)]

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

                    H_user = H[(b, k)]

                    # 构建干扰矩阵
                    A_m = 1e-10 * np.eye(Mt, dtype=complex)
                    for v in BS_serve[b]:
                        if v != k:
                            H_j = H[(b, v)]
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
                        ICI_mean_estimate[(k, l, b, v)] = 1 / Mt * (
                                    (v_all[(k, l)].conj().T @ Rr[(b, k)] @ v_all[(k, l)]) * np.trace(
                                Rt[(b, v)] @ Rt[(b, k)])).real
                        ICI_mean[(k, l, b, v)] = 0
                        for i in range(n_layer):
                            precoder = precoder_all[v][:, i]
                            precoder /= np.linalg.norm(precoder)
                            ICI_mean[(k, l, b, v)] += (combiner @ Rr[(b, k)] @ combiner.conj().T).real * (
                                        precoder.conj().T @ Rt[(b, k)] @ precoder).real
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
