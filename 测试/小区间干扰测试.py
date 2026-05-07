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
noise0 = 10 ** (-174/10 - 3)              # 噪声功率(W/Hz)
noise = noise0 * 30 * 1e3 * 12
P = 10 ** (23.18/10 - 3)
WMMSE_max_iteration = 5

quadriga_channel = True

if not quadriga_channel:

    def get_largescale(dub, fc):
        beta_dB = 31.84 + 21.5 * np.log10(dub) + 19 * np.log10(fc * 1e-3)
        return 10 ** (- beta_dB / 10)

    # 统计ICI
    ICI = []
    samples = 10000   #随机实现次数

    for sample in range(samples):
        print(f"now running sample: {sample}")
        # 随机初始化用户位置
        User_pos = []
        BS_serve = {}
        for b in range(B):
            BS_serve[b] = set()
            for k in range(K * B):
                if k // 2 == b:
                    User_pos.append((BS_pos[b][0] + (np.random.rand() - 0.5) * region_bound, BS_pos[b][1] + (np.random.rand() - 0.5) * region_bound))
                    BS_serve[b].add(k)

        # 生成大尺度与小尺度衰落
        H = {}
        Hl = {}
        Hs = {}
        for b in range(B):
            for k in range(K * B):
                dub = get_dis(User_pos[k], BS_pos[b], h_U, h_BS)
                Hl[(b, k)] = np.sqrt(get_largescale(dub, fc))
                # 生成标准复高斯分布数据
                real_part = np.random.normal(0, 0.5, size=(Mt, Mr))  # 实部
                imaginary_part = np.random.normal(0, 0.5, size=(Mt, Mr))  # 虚部
                # 得到小尺度衰落
                Hs[(b, k)] = real_part + 1j * imaginary_part
                H[(b, k)] = Hl[(b, k)] * Hs[(b, k)]

        # 各个基站做预编码
        precoder_all = {}
        combiner_all = {}
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
                            ICI.append(np.linalg.norm(combiner @ Hs[(b, k)].conj().T @ precoder) ** 2)

    num_bins = 1000  # 直方图的柱子数量
    bin_range = (0, 0.2)  # 数据范围

    # 绘制直方图
    n, bins, patches = plt.hist(ICI, num_bins, bin_range, density=True, alpha=0.6, color='g')

    # 设置图表标签和标题
    plt.xlabel('ICI')
    plt.ylabel('Frequency')
    plt.title('ICI Frequency Distribution Histogram')

    # 显示图表
    plt.show()

else:

    '''
        如果要生成带小区间干扰的QuaDriGa信道则需要大概信道生成的matlab代码
        这里只是想测试一下算法效果，所以依旧采用quadriga_UE_packs_test文件夹的信道
        这一包中信道原本用于训练，一个pack包含480TTI连续信道，不同包之间信道独立
        考虑到这里仿真时需要两个cell、每个cell里两个用户，需要各个BS对自己两用户的信道和对另一BS两用户的干扰信道
        因此采用4个pack来分别模拟，第一个pack作为BS 0的两用户，第二个pack作为BS 1的两用户，
        第三个pack作为BS 0到BS 1的干扰信道，第四个pack作为BS 1到BS 0的干扰信道
    '''

    from 基线测试_quadriga import QuadrigaChannelSource

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

    # quadriga信道文件地址
    quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/quadriga_UE_packs_test"
    qg_src = QuadrigaChannelSource(quadriga_dir)

    pack_len = 480
    start_pack = pack_len*4
    # 统计ICI
    ICI = []

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

        for b in range(B):
            for k in range(K * B):
                if k in BS_serve[b]:
                    slots = start_pack + pack_len * b + sample
                else:
                    slots = start_pack + pack_len * b + pack_len * 2 + sample
                H[(b, k)] = qg_src.get_H(k % 2, slots)      # [NRB, NBS, N_RX]
                assert (H[(b, k)].shape[1], H[(b, k)].shape[2]) == (Mt, Mr), "channel shape mismatch!"
                # 得到小尺度衰落
                Hl[(b, k)], Hs[(b, k)] = separate_large_scale_fading(H[(b, k)])

        # 各个基站做预编码
        precoder_all = {}
        combiner_all = {}
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
                            ICI.append(np.linalg.norm(combiner @ Hs[(b, k)][0].conj().T @ precoder) ** 2)

    num_bins = 1000  # 直方图的柱子数量
    bin_range = (0, 2)  # 数据范围
    plt.ylim(0, 2)

    # 绘制直方图
    n, bins, patches = plt.hist(ICI, num_bins, bin_range, density=True, alpha=0.6, color='g')

    # 设置图表标签和标题
    plt.xlabel('ICI')
    plt.ylabel('Frequency')
    plt.title(f'ICI Frequency Distribution Histogram {start_pack // pack_len}')

    # 显示图表
    plt.show()

# # test
# sinr_list = []
# sinr_list_dB = []
# up_list = []
# down_list = []
# for l in range(n_layer):
#     up = 0
#     down = 0
#     combiner = combiner_all[(0, 0)][l, :]
#
#     for v in BS_serve[0]:
#         for i in range(n_layer):
#             precoding_vector = precoder_all[(0, v)][:, i]
#
#             if v == 0 and i == l:
#                 up = np.linalg.norm(combiner @ H[(0, 0)].conj().T @ precoding_vector) ** 2
#             else:
#                 down += np.linalg.norm(combiner @ H[(0, 0)].conj().T @ precoding_vector) ** 2
#
#     sinr_list.append(up / (down + noise))
#     sinr_list_dB.append(10 * np.log10(up / (down + noise)))
#     up_list.append(up)
#     down_list.append(down)
# sinr = np.exp(np.mean(np.log(np.array(sinr_list))))  # 层间几何平均
# sinr = 10 * np.log10(sinr)
# print(sinr)