import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_channel(R, total_samples):
    """
    高效批量生成复高斯随机向量

    Args:
        R: 协方差矩阵 (M, M)

    Returns:
        形状为 (batch_size, M) 的数组
    """
    M = R.shape[0]
    # L = np.linalg.cholesky(R)

    # 步骤1：特征值分解（对 Hermitian 矩阵用 eigh，确保数值稳定）
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # 步骤2：处理特征值（确保非负，避免数值误差导致的微小负数）
    epsilon = 1e-20
    eigenvalues = np.maximum(eigenvalues, epsilon)  # 替换负特征值为极小正数
    lambda_sqrt = np.sqrt(eigenvalues)  # 特征值开平方

    # 步骤3：构造平方根矩阵 L = U * Lambda^(1/2)
    L = eigenvectors @ np.diag(lambda_sqrt)

    # 一次性生成所有随机数
    h_real = np.random.randn(total_samples, M)
    h_imag = np.random.randn(total_samples, M)
    h0 = (h_real + 1j * h_imag) / np.sqrt(2)

    # 批量变换
    h = (L @ h0.T).T

    # 重塑为批次格式
    return h.reshape(total_samples, M), L


M = 256          # 天线数
# r = 0.2 * np.exp(-1j * np.pi/4)         # 信道空间相关系数
theta_bar = np.pi/2                       # 固定信道中心角，从ULA平面方向算起
delta_theta = 1/180 * np.pi              # 围绕中心角向两边扩散的最大角度  原2.5
kai = np.pi                               # 2*pi*d/λ，固定取d/λ为1/2
T = 1000000       # 蒙特卡洛仿真次数
N = 256           # 阵列流形矢量角度颗粒度
trajectory = 10000                        # 老化轨迹蒙特卡洛仿真数量
period = 160                              # 老化TTI数量
rho = 0.9966                              # 一阶AR时间相关
rho2 = np.sqrt(1 - rho**2)


# 生成信道空间相关矩阵
R = np.zeros((M, M), dtype=np.complex128)
for m in range(M):
    for n in range(M):
        R[m][n] = 1/M * np.exp(-1j * kai*(m-n) * np.cos(theta_bar)) * np.sinc(kai*(m-n) * delta_theta * np.sin(theta_bar))

# 生成用于测试的阵列流形矢量矩阵
a = np.zeros((N, M), dtype=complex)
theta = [np.pi / (N - 1) * i for i in range(N)]
# 构建N * M的阵列流形矢量矩阵，其中第i列为指向第i个方向的阵列流形矢量
for i, t in enumerate(theta):
    a[i, :] = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * np.cos(t))

# 生成老化信道
aged_channel = []
channel, L = generate_gaussian_channel(R, trajectory)
for t in range(period):
    aged_channel.append(channel)
    g, _ = generate_gaussian_channel(R, trajectory)
    channel = channel * rho + g * rho2

# # 测试信道角度分布情况
# channels = generate_gaussian_channel(R, T)
#
# ave_angel_response = np.abs(channels @ a.T)
# ave_angel_response = np.mean(ave_angel_response, axis=0)
#
# plt.figure()
# plt.plot([t/np.pi for t in theta], ave_angel_response)
# plt.grid(True)
# plt.title(f"channel angel response")
# plt.xlabel("angle/Π")
# plt.legend()
# plt.show()


# 生成DFT码本，等效于在0到Π上均匀分布的角度
# temp = np.outer(np.arange(M), np.arange(M))
DFT_codebook = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)
idx = np.arange(M) / M * 2
idx = [i if i <= 1 else i-2 for i in idx]
angle = np.arccos(idx) / np.pi
seq_angle = np.arange(M)/M

# # 基线：DFT码本
# # 这里选择固定DFT码字的形式，既然已知中心角为Π/2，就直接选择0号码字
# F = DFT_codebook[:, 0]
#
# DFT_angel_response = np.abs(a.conj() @ F)
# plt.figure()
# plt.plot([t/np.pi for t in theta], DFT_angel_response)
# plt.grid(True)
# plt.title(f"DFT angel response")
# plt.xlabel("angle/Π")
# plt.show()
#
# # 测试这一码字对信道老化的影响，方式为生成大量老化轨迹，对轨迹取平均
# gain = []
# for t in range(period):
#     gain.append(10 * np.log10(np.mean(np.abs(aged_channel[t] @ F.conj()), axis=0)))
# plt.figure()
# plt.plot(gain)
# plt.grid(True)
# plt.title(f"DFT gain on aged channel")
# plt.xlabel("TTI")
# plt.ylabel("dB")
# plt.show()

# # 基线：DFT码本
# # 这里选择动态DFT码字的形式，在轨迹开始时选择最优DFT码字，之后整条轨迹上不变
# F = np.zeros((M, trajectory), dtype=np.complex128)
# for i in range(trajectory):
#     idx = np.argmax(np.abs(aged_channel[0][i, :] @ DFT_codebook.conj()))
#     F[:, i] = DFT_codebook[:, idx]
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         gain_sum += np.abs(aged_channel[t][i, :] @ F[:, i].conj())
#     gain.append(gain_sum/trajectory)
# plt.figure()
# plt.plot(gain, label='DFT')
# plt.grid(True)
# # plt.title(f"DFT gain on aged channel: adapt codeword")
# # plt.xlabel("TTI")
# # plt.ylabel("dB")
# # plt.show()
#
# 基线：MRT
# 单用户情形下最优预编码为MRT
gain = []
F = aged_channel[0].conj().T
for t in range(period):
    gain_sum = 0
    for i in range(trajectory):
        gain_sum += np.abs(aged_channel[t][i, :] @ F[:, i] / np.linalg.norm(F[:, i]))
    gain.append(gain_sum/trajectory)
# plt.figure()
plt.plot(gain, label='MRT')
# plt.grid(True)
# plt.title(f"MRT gain on aged channel")
# plt.xlabel("TTI")
# plt.ylabel("dB")
# plt.show()
#
# # 基线：外层DFT码本+内层MRT
# # 外层选择3个最适配的DFT码字，构建成等效信道，内层在等效信道上进行MRT
# # precoder功率需要归一化
# F = np.zeros((M, trajectory), dtype=np.complex128)
# for i in range(trajectory):
#     sorted_idx = np.argsort(np.abs(aged_channel[0][i, :] @ DFT_codebook.conj()))[::-1]
#     temp = DFT_codebook[:, sorted_idx[0:3]]
#     F[:, i] = temp @ (aged_channel[0][i, :].conj() @ temp).conj().T
#     F[:, i] = F[:, i] / np.linalg.norm(F[:, i])
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         gain_sum += np.abs(aged_channel[t][i, :].conj() @ F[:, i])
#     gain.append(gain_sum/trajectory)
# # plt.figure()
# plt.plot(gain, label='DFT + MRT')
# # plt.grid(True)
# # plt.title(f"double precoder DFT + MRT gain on aged channel")
# # plt.xlabel("TTI")
# # plt.ylabel("dB")
# # plt.show()

# # 方案1：宽波束设计
# # 将宽波束看作类似方波的造型，直接面向结果来设计预编码。
# A_BS_D = a.T.copy()     # 论文中M * N的指向各个方向的阵列流形矢量
# G = np.zeros((N, 1))    # 指示在哪些位置需要有增益。这里与论文不同，直接面向角度来设计，认为我已知信道的角度分布情况
# # 生成的信道在角度上的功率分布约为Π/2左右各Π/20，设计宽波束时也将这一范围覆盖，取114~141号流形方向
# G[114:141] = 1
#
# F = np.linalg.inv(A_BS_D @ A_BS_D.conj().T) @ A_BS_D @ G
# F = F / np.linalg.norm(F)
#
# wide_beam_angel_response = np.abs(a.conj() @ F)
# plt.figure()
# plt.plot([t/np.pi for t in theta], wide_beam_angel_response)
# plt.grid(True)
# plt.title(f"wide beam angel response")
# plt.xlabel("angle/Π")
# plt.show()
#
# # 测试这一码字对信道老化的影响，方式为生成大量老化轨迹，对轨迹取平均
# gain = []
# for t in range(period):
#     gain.append(10 * np.log10(np.mean(np.abs(aged_channel[t] @ F.conj()), axis=0)))
# plt.figure()
# plt.plot(gain)
# plt.grid(True)
# plt.title(f"wide beam gain on aged channel")
# plt.xlabel("TTI")
# plt.ylabel("dB")
# plt.show()

# # 方案1对比：波束稍窄一点
# A_BS_D = a.T.copy()     # 论文中M * N的指向各个方向的阵列流形矢量
# G = np.zeros((N, 1))    # 指示在哪些位置需要有增益。这里与论文不同，直接面向角度来设计，认为我已知信道的角度分布情况
# # 生成的信道在角度上的功率分布约为Π/2左右各Π/20，作为对比将波束取得更窄，取120~135号流形方向
# G[120:135] = 1
#
# F = np.linalg.inv(A_BS_D @ A_BS_D.conj().T) @ A_BS_D @ G
# F = F / np.linalg.norm(F)
#
# wide_beam_angel_response = np.abs(a.conj() @ F)
# plt.figure()
# plt.plot([t/np.pi for t in theta], wide_beam_angel_response)
# plt.grid(True)
# plt.title(f"wide beam angel response: narrower")
# plt.xlabel("angle/Π")
# plt.show()
#
# # 测试这一码字对信道老化的影响，方式为生成大量老化轨迹，对轨迹取平均
# gain = []
# for t in range(period):
#     gain.append(10 * np.log10(np.mean(np.abs(aged_channel[t] @ F.conj()), axis=0)))
# plt.figure()
# plt.plot(gain)
# plt.grid(True)
# plt.title(f"wide beam gain on aged channel: narrower")
# plt.xlabel("TTI")
# plt.ylabel("dB")
# plt.show()

# # 方案1：宽波束中心适应
# # 对方案1宽波束再优化一下，用阵列流形矢量匹配信道实现的最优角度，再以那一角度为中心展开宽波束设计
# A_BS_D = a.T.copy()     # 论文中M * N的指向各个方向的阵列流形矢量
# F = np.zeros((M, trajectory), dtype=np.complex128)
# for i in range(trajectory):
#     idx = np.argmax(np.abs(aged_channel[0][i, :] @ a.conj().T))
#     # 这里将最适配的角度左右展开为宽波束
#     G = np.zeros((N, 1))  # 指示在哪些位置需要有增益
#     G[idx-7:idx+8] = 1
#     F[:, i] = (np.linalg.inv(A_BS_D @ A_BS_D.conj().T) @ A_BS_D @ G).ravel()
#     F[:, i] = F[:, i] / np.linalg.norm(F[:, i])
#
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         gain_sum += np.abs(aged_channel[t][i, :] @ F[:, i].conj())
#     gain.append(gain_sum/trajectory)
# # plt.figure()
# plt.plot(gain, label='wide beam')
# # plt.grid(True)
# # plt.title(f"wide beam gain on aged channel: adapt angle")
# # plt.xlabel("TTI")
# # plt.ylabel("dB")
# # plt.show()
#
# # 方案2：DFT组合宽波束
# # 宽波束由DFT码字组合而成
# # 先前的测试表明这一组合方式未必十分优秀，但胜在比较灵活
# # 这里先使用DFT码本确定方向，再将周边的DFT码字组合形成宽波束
# # 实际中如果使用这一方案，则可以考虑波束的宽窄变化，这里只考察宽波束下的SNR老化情况
# # 或者结合阵列流形矢量矩阵，先估计出方向情况，再设计宽波束？
# DFT_codebook_temp = 1 / np.sqrt(M) * np.exp(
#     2j * np.pi * np.outer(np.arange(M), (np.arange(M) + (1-M)/2)) / M)
# F = np.zeros((M, trajectory), dtype=np.complex128)
# for i in range(trajectory):
#     idx = np.argmax(np.abs(aged_channel[0][i, :] @ DFT_codebook_temp.conj()))
#     # 这里将最适配的DFT码字及左右各一个码字组合起来
#     if idx == 31:
#         test = [DFT_codebook_temp[:, j] * np.exp(1j * np.pi * (-1 + 1 / M) * (j + 1)) for j in range(idx - 1, idx + 1)]
#     elif idx == 0:
#         test = [DFT_codebook_temp[:, j] * np.exp(1j * np.pi * (-1 + 1 / M) * (j + 1)) for j in range(idx, idx+2)]
#     else:
#         test = [DFT_codebook_temp[:, j] * np.exp(1j * np.pi * (-1 + 1 / M) * (j + 1)) for j in range(idx - 1, idx + 2)]
#     F[:, i] = np.sum(test, axis=0)
#     F[:, i] = F[:, i] / np.linalg.norm(F[:, i])
#
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         gain_sum += np.abs(aged_channel[t][i, :] @ F[:, i].conj())
#     gain.append(gain_sum/trajectory)
# # plt.figure()
# plt.plot(gain, label='wide beam (DFT combine)')
# # plt.grid(True)
# # plt.title(f"wide beam gain on aged channel: DFT combine")
# # plt.xlabel("TTI")
# # plt.ylabel("dB")
# # plt.legend()
# # plt.title("beam gain of different precoder on aged channel")
# # plt.show()
#
# # 方案2对比：外层DFT组合宽波束 + 内层MRT
# # 这一方案有一潜在问题，即宽波束之间是否允许重叠？
# DFT_codebook_temp = 1 / np.sqrt(M) * np.exp(
#     2j * np.pi * np.outer(np.arange(M), (np.arange(M) + (1-M)/2)) / M)
# F = np.zeros((M, trajectory), dtype=np.complex128)
# for i in range(trajectory):
#     sorted_idx = np.argsort(np.abs(aged_channel[0][i, :].conj() @ DFT_codebook_temp))[::-1]
#     temp = DFT_codebook[:, sorted_idx[0:3]]
#     for k, idx in enumerate(sorted_idx[0:3]):
#         # 这里将最适配的DFT码字及左右各一个码字组合起来
#         if idx == 31:
#             temp[:, k] = np.sum([DFT_codebook_temp[:, j] * np.exp(1j * np.pi * (-1 + 1 / M) * (j + 1)) for j in range(idx-1, idx+1)], axis=0)
#         elif idx == 0:
#             temp[:, k] = np.sum([DFT_codebook_temp[:, j] * np.exp(1j * np.pi * (-1 + 1 / M) * (j + 1)) for j in range(idx, idx+2)], axis=0)
#         else:
#             temp[:, k] = np.sum([DFT_codebook_temp[:, j] * np.exp(1j * np.pi * (-1 + 1 / M) * (j + 1)) for j in range(idx-1, idx+2)], axis=0)
#     F[:, i] = temp @ (aged_channel[0][i, :].conj() @ temp).conj().T
#     F[:, i] = F[:, i] / np.linalg.norm(F[:, i])
#
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         gain_sum += np.abs(aged_channel[t][i, :].conj() @ F[:, i])
#     gain.append(gain_sum/trajectory)
# # plt.figure()
# plt.plot(gain, label='wide beam (DFT combine) + MRT')
# # plt.grid(True)
# # plt.title(f"double precoder wide beam + MRT gain on aged channel: DFT combine")
# # plt.xlabel("TTI")
# # plt.ylabel("dB")
# # plt.legend()
# # plt.title("beam gain of different precoder on aged channel")
# # plt.show()
#
# # 方案3：DFT波束动态选择
# # 参考论文公式，基于信道相关矩阵分析在统计意义上最可能选中的码字，再结合时间相关系数，从瞬时增益过渡到到统计数值指导码字选择
# # F = np.zeros((M, trajectory), dtype=np.complex128)
# aged_gain = np.zeros((trajectory, M), dtype=np.complex128)
# rho_step = 1
# rho2_step = 0
# statistical_gain = np.zeros((M,), dtype=np.complex128)
# for i in range(trajectory):
#     aged_gain[i, :] = np.abs(aged_channel[0][i, :].conj() @ DFT_codebook) ** 2
# for i in range(M):
#     statistical_gain[i] = (rho2 * np.linalg.norm(L @ DFT_codebook[:, i])) ** 2
#
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         code_gain = (rho_step ** 2) * aged_gain[i, :] + rho2_step * statistical_gain
#         idx = np.argmax(code_gain)
#         gain_sum += np.abs(aged_channel[t][i, :].conj() @ DFT_codebook[:, idx])
#     rho2_step += rho_step
#     rho_step = rho_step * rho
#     gain.append(gain_sum/trajectory)
# # plt.figure()
# plt.plot(gain, label='dynamic DFT', linestyle='--')
# # plt.grid(True)
# # plt.title(f"dynamic DFT gain on aged channel")
# # plt.xlabel("TTI")
# # plt.ylabel("dB")
# # plt.legend()
# # plt.title("beam gain of different precoder on aged channel")
# # plt.show()
#
# # 方案3对比：DFT波束动态选择 + MRT
# # 外层动态选择3个波束，内层MRT
# aged_gain = np.zeros((trajectory, M), dtype=np.complex128)
# rho_step = 1
# rho2_step = 0
# statistical_gain = np.zeros((M,), dtype=np.complex128)
# for i in range(trajectory):
#     aged_gain[i, :] = np.abs(aged_channel[0][i, :].conj() @ DFT_codebook) ** 2
# for i in range(M):
#     statistical_gain[i] = (rho2 * np.linalg.norm(L @ DFT_codebook[:, i])) ** 2
#
# gain = []
# for t in range(period):
#     gain_sum = 0
#     for i in range(trajectory):
#         code_gain = (rho_step ** 2) * aged_gain[i, :] + rho2_step * statistical_gain
#         sorted_idx = np.argsort(code_gain)[::-1]
#         temp = DFT_codebook[:, sorted_idx[0:3]]
#         F = temp @ (aged_channel[0][i, :].conj() @ temp).conj().T
#         F = F / np.linalg.norm(F)
#         gain_sum += np.abs(aged_channel[t][i, :].conj() @ F)
#     rho2_step += rho_step
#     rho_step = rho_step * rho
#     gain.append(gain_sum/trajectory)
# # plt.figure()
# plt.plot(gain, label='dynamic DFT + MRT', linestyle='--')
# # plt.grid(True)
# # plt.title(f"double precoder dynamic DFT + MRT gain on aged channel")
# plt.xlabel("TTI")
# plt.ylabel("dB")
# plt.legend()
# plt.title("beam gain of different precoder on aged channel")
# plt.show()


# 测试类JSDM预编码方案
# channel, L, h0 = generate_gaussian_channel(R, 1)

eigenvalues, eigenvectors = np.linalg.eigh(R)

# 获取特征值从大到小的索引
sorted_indices = np.argsort(eigenvalues.real)[::-1]

# 按特征值大小排序
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

rank = np.sum(np.abs(sorted_eigenvalues) > 1e-6)

# M = min(rank, self.N_rf)
V = sorted_eigenvectors[:, :rank]

F = aged_channel[0].T
for i in range(trajectory):
    F[:, i] = V @ (aged_channel[0][i, :].conj() @ V).conj().T

gain = []
for t in range(period):
    gain_sum = 0
    for i in range(trajectory):
        gain_sum += aged_channel[t][i, :].conj() @ F[:, i]
    gain.append(gain_sum/trajectory)
# plt.figure()
plt.plot(gain, label='JSDM-like', linestyle='--')
plt.grid(True)
# plt.title(f"double precoder JSDM-like + MRT gain on aged channel")
plt.xlabel("TTI")
plt.ylabel("dB")
plt.legend()
plt.title("beam gain of different precoder on aged channel")
plt.show()

# # 测试层次码本码字组合后的响应情况
# DFT_codebook_temp = 1 / np.sqrt(M) * np.exp(
#     2j * np.pi * np.outer(np.arange(M), (np.arange(M) + (1-M)/2)) / M)
# DFT_codebook2 = np.zeros((M, int(M/2)), dtype=np.complex128)
# for i in range(0, M, 2):
#     idx = i // 2
#     DFT_codebook2[:, idx] = DFT_codebook_temp[:, i] * np.exp(1j * np.pi * (-1 + 1/M) * (i+1)) + DFT_codebook_temp[:, i+1] * np.exp(1j * np.pi * (-1 + 1/M) * (i+2))
# DFT_codebook2 = 1/np.sqrt(2) * DFT_codebook2.conj()
#
# test = [DFT_codebook_temp[:, i] * np.exp(1j * np.pi * (-1 + 1/M) * (i+1)) for i in range(16, 19)]
# DFT_combine_4 = np.sum(test, axis=0) * 1/2
# DFT_combine_4 = DFT_combine_4.conj()
#
# # print(DFT_combine_4.conj().T @ DFT_combine_4)
#
# L = 200
# a= np.zeros((L,M), dtype=complex)
# theta = [np.pi/(L-1) * i for i in range(L)]
# for i, t in enumerate(theta):
#     a[i, :] = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * np.cos(t))
#
# angel_response1 = np.abs(a.conj() @ DFT_codebook_temp[:, 16].conj())
# angel_response12 = np.abs(a.conj() @ DFT_codebook_temp[:, 17].conj())
# angel_response13 = np.abs(a.conj() @ DFT_codebook_temp[:, 18].conj())
# # angel_response14 = np.abs(a.conj() @ DFT_codebook_temp[:, 19].conj())
# # angel_response2 = np.abs(a.conj() @ DFT_codebook2[:, 8])
# # angel_response21 = np.abs(a.conj() @ DFT_codebook2[:, 9])
# # angel_response22 = np.abs(a.conj() @ DFT_codebook2[:, 10])
# angel_response24 = np.abs(a.conj() @ DFT_combine_4)
#
# plt.figure()
# plt.plot([t/np.pi for t in theta], angel_response1, label="DFT")
# plt.plot([t/np.pi for t in theta], angel_response12)
# plt.plot([t/np.pi for t in theta], angel_response13)
# # plt.plot([t/np.pi for t in theta], angel_response14)
# plt.plot([t/np.pi for t in theta], angel_response24, label="combine")
# # plt.plot([t/np.pi for t in theta], angel_response21)
# plt.plot([t/np.pi for t in theta], angel_response22)
# plt.grid(True)
# plt.title(f"codebook angel response")
# plt.xlabel("angle/Π")
# plt.legend()
# plt.show()

# # 仿真信道角度响应情况
# channels = generate_gaussian_channel(R, T)
# ave_angel_response = np.abs(channels @ DFT_codebook)
# ave_angel_response = np.mean(ave_angel_response, axis=0)
#
# result = ave_angel_response[::-1].copy()
# result[0:int(M/2)+1], result[int(M/2)+1:M] = ave_angel_response[::-1][int(M/2)-1:M], ave_angel_response[::-1][0:int(M/2)-1]
#
# plt.figure()
# plt.plot(seq_angle, result)
# plt.grid(True)
# plt.title(f"average angel response")
# plt.xlabel("angle/Π")
# plt.show()
#
# # 作为对比，生成一个单径信道
# a = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2)
# small_scale = (np.random.randn(T, 1) + 1j * np.random.randn(T, 1)) / np.sqrt(2)
# ave_angel_response = np.abs((small_scale @ a.reshape(1, M)).conj() @ DFT_codebook)
# ave_angel_response = np.mean(ave_angel_response, axis=0)
#
# result = ave_angel_response[::-1].copy()
# result[0:int(M/2)+1], result[int(M/2)+1:M] = ave_angel_response[::-1][int(M/2)-1:M], ave_angel_response[::-1][0:int(M/2)-1]
#
# plt.figure()
# plt.plot(seq_angle, result)
# plt.grid(True)
# plt.title(f"average angel response (one path)")
# plt.xlabel("angle/Π")
# plt.show()
#
# # 作为对比，生成一个三径合成的信道
# theta = [0, np.pi/6, np.pi/3]
# a= np.zeros((3,M), dtype=complex)
# for i, t in enumerate(theta):
#     a[i, :] = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * np.cos(t))
# small_scale = (np.random.randn(T, 3) + 1j * np.random.randn(T, 3)) / np.sqrt(2)
# ave_angel_response = np.abs((small_scale @ a).conj() @ DFT_codebook)
# ave_angel_response = np.mean(ave_angel_response, axis=0)
#
# result = ave_angel_response[::-1].copy()
# result[0:int(M/2)+1], result[int(M/2)+1:M] = ave_angel_response[::-1][int(M/2)-1:M], ave_angel_response[::-1][0:int(M/2)-1]
#
# plt.figure()
# plt.plot(seq_angle, result)
# plt.grid(True)
# plt.title(f"average angel response (three paths)")
# plt.xlabel("angle/Π")
# plt.show()

# # 作为对比，生成一个单径信道，角度特殊
# a = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * np.cos(np.pi/2 * 0.8))
# small_scale = (np.random.randn(T, 1) + 1j * np.random.randn(T, 1)) / np.sqrt(2)
# ave_angel_response = np.abs((small_scale @ a.reshape(1, M)).conj() @ DFT_codebook)
# ave_angel_response = np.mean(ave_angel_response, axis=0)
#
# plt.figure()
# plt.plot(ave_angel_response)
# plt.grid(True)
# plt.title(f"average angel response (one path, 0.4Π)")
# plt.show()
#
# # 绘制DFT码本码字编号到物理角度的映射图
# idx = np.arange(M) / M * 2
# idx = [i if i <= 1 else i-2 for i in idx]
# angle = np.arccos(idx) / np.pi
# plt.figure()
# plt.plot(angle)
# plt.grid(True)
# plt.title("DFT idx to angle map")
# plt.show()
#
# # 作为对比，生成一个多径合成的信道
# L = 200
# a= np.zeros((L,M), dtype=complex)
# # theta = [i/L * np.pi/8 + 7*np.pi/16 for i in range(L)]
# theta = [np.pi/(L-1) * i for i in range(L)]
# for i, t in enumerate(theta):
#     a[i, :] = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * np.cos(t))
#     # a[i, :] = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * t)
# small_scale = (np.random.randn(T, L) + 1j * np.random.randn(T, L)) / np.sqrt(2)
# ave_angel_response = np.abs((small_scale @ a).conj() @ DFT_codebook)
# ave_angel_response = np.mean(ave_angel_response, axis=0)
#
# result = ave_angel_response[::-1].copy()
# result[0:int(M/2)+1], result[int(M/2)+1:M] = ave_angel_response[::-1][int(M/2)-1:M], ave_angel_response[::-1][0:int(M/2)-1]
#
# plt.figure()
# plt.plot(seq_angle, result)
# plt.grid(True)
# plt.title(f"average angel response ({L} paths)")
# plt.xlabel("angle/Π")
# plt.show()