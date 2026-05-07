import numpy as np
import matplotlib.pyplot as plt

from common.tools import get_dis


# # 测试不同用户分布情况下估计结果是否鲁棒
# def generate_gaussian_channel(R):
#
#     M = R.shape[1]
#     K = R.shape[0]
#     h = np.zeros((K, M), dtype=np.complex128)
#
#     for k in range(K):
#         # 步骤1：特征值分解（对 Hermitian 矩阵用 eigh，确保数值稳定）
#         eigenvalues, eigenvectors = np.linalg.eigh(R[k])
#
#         # 步骤2：处理特征值（确保非负，避免数值误差导致的微小负数）
#         epsilon = 1e-18
#         eigenvalues = np.maximum(eigenvalues, epsilon)  # 替换负特征值为极小正数
#         lambda_sqrt = np.sqrt(eigenvalues)  # 特征值开平方
#
#         # 步骤3：构造平方根矩阵 L = U * Lambda^(1/2)
#         L = eigenvectors @ np.diag(lambda_sqrt)
#
#         # 一次性生成所有随机数
#         h_real = np.random.randn(1, M)
#         h_imag = np.random.randn(1, M)
#         h0 = (h_real + 1j * h_imag) / np.sqrt(2)
#
#         # 批量变换
#         h[k, :] = (L @ h0.T).T
#
#     return h
#
# def calculate_AOD(x1, y1, x2, y2):
#     dx, dy = x2 - x1, y2 - y1
#     theta_bar = np.arctan2(dx, dy)
#     return np.abs(theta_bar)
#
# def get_largescale(dub, fc):
#
#     beta_dB = 31.84 + 21.5 * np.log10(dub) + 19 * np.log10(fc * 1e-3)
#
#     return 10 ** (- beta_dB/10)
#
#
# K = 2
# M = 256
# kai = np.pi
# delta_theta = 5/180 * np.pi
# BS_pos = [300, 300]
# u_pos = [[[ 52, 201 ], [ 178, 262 ]], [[ 52, 201 ], [ 178, 62 ]], [[ 52, 201 ], [ 178, 562 ]]]
# h_U = 1.65
# h_BS = 15
# fc = 3500
# noise = 10 ** (-142/10 - 3) * 30 * 1e3
# P0 = 10 ** (10/10 - 3)
#
# T = 160
# N = 50
# I = 0.013
# rho = 0.9966
#
# for pos_idx in range(3):
#     # 生成信道空间相关矩阵
#     R = np.zeros((K, M, M), dtype=np.complex128)
#     for k in range(K):
#         for m in range(M):
#             for n in range(M):
#                 theta_bar = calculate_AOD(BS_pos[0], BS_pos[1], u_pos[pos_idx][k][0], u_pos[pos_idx][k][1])
#                 R[k][m][n] = 1 / M * np.exp(
#                     -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                     kai * (m - n) * delta_theta * np.sin(theta_bar))
#
#     Hl = np.zeros((K, M))
#     for k in range(K):
#         dub = get_dis(u_pos[pos_idx][k], BS_pos, h_U, h_BS)
#         Hl[k, :] = np.sqrt(get_largescale(dub, fc))
#     Hs = generate_gaussian_channel(R)
#     H = Hl * Hs
#
#     # 用基本的一套信道生成一次MMSE预编码，并记录功率
#     inv = np.linalg.inv(H.conj() @ H.T + K * noise / P0 * np.eye(K))
#     precoder = H.T @ inv
#     precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#     P = []
#     for k in range(K):
#         P.append(np.linalg.norm(precoder[:, k]) ** 2)
#
#     gain = []
#     interference = []
#
#     for k in range(K):
#         i = k + 1 if k == 0 else 0
#         gain.append((Hl[k, 0] ** 2) * P[k])
#         interference.append((Hl[k, 0] ** 2) * P[i] * np.trace(R[i, :, :] @ R[k, :, :]).real)
#
#     sinr_estimate = [[] for _ in range(K)]
#     gain_estimate = [[] for _ in range(K)]
#     interference_estimate = [[] for _ in range(K)]
#     for t in range(T):
#         sinr_estimate[0].append(10 * np.log10((gain[0] * (rho ** (2 * t))) / (interference[0] * (1 - rho ** (2 * t)) + noise)))
#         sinr_estimate[1].append(10 * np.log10((gain[1] * (rho ** (2 * t))) / (interference[1] * (1 - rho ** (2 * t)) + noise)))
#         gain_estimate[0].append(10 * np.log10(gain[0] * (rho ** (2 * t))))
#         gain_estimate[1].append(10 * np.log10(gain[1] * (rho ** (2 * t))))
#         interference_estimate[0].append(10 * np.log10(interference[0] * (1 - rho ** (2 * t)) + 1e-20))
#         interference_estimate[1].append(10 * np.log10(interference[1] * (1 - rho ** (2 * t)) + 1e-20))
#
#     # 蒙特卡洛仿真实际信道老化情况
#     gain_real = [[0] * T for _ in range(K)]
#     interference_real = [[0] * T for _ in range(K)]
#     sinr_real = [[0] * T for _ in range(K)]
#     for n in range(N):
#         print(f"n = {n}")
#         Hs_m = generate_gaussian_channel(R)
#         H_m = Hl * Hs_m
#         inv = np.linalg.inv(H_m.conj() @ H_m.T + K * noise / P0 * np.eye(K))
#         precoder = H_m.T @ inv
#         precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#         gain_temp = [0] * K
#         interference_temp = [0] * K
#         for t in range(T):
#             for k in range(K):
#                 i = k + 1 if k == 0 else 0
#                 gain_temp[k] = np.linalg.norm(H_m[k, :].conj() @ precoder[:, k]) ** 2
#                 gain_real[k][t] += 10 * np.log10(gain_temp[k])
#                 interference_temp[k] = np.linalg.norm(H_m[k, :].conj() @ precoder[:, i]) ** 2
#                 interference_real[k][t] += 10 * np.log10(interference_temp[k])
#                 sinr_real[k][t] += 10 * np.log10(gain_temp[k] / (interference_temp[k] + noise))
#             G = generate_gaussian_channel(R)
#             Hs_m = rho * Hs_m + np.sqrt(1 - rho ** 2) * G
#             H_m = Hl * Hs_m
#     for t in range(T):
#         for k in range(K):
#             gain_real[k][t] /= N
#             interference_real[k][t] /= N
#             sinr_real[k][t] /= N
#
#     plt.figure()
#     plt.plot(sinr_estimate[0], label='user 0 sinr estimate', linestyle='--')
#     plt.plot(sinr_estimate[1], label='user 1 sinr estimate', linestyle='--')
#     for k in range(K):
#         plt.plot(sinr_real[k], label=f"user {k} ave sinr")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'sinr simulation {pos_idx}')
#     plt.show()
#
#     plt.figure()
#     plt.plot(gain_estimate[0], label='user 0 gain estimate', linestyle='--')
#     plt.plot(gain_estimate[1], label='user 1 gain estimate', linestyle='--')
#     for k in range(K):
#         plt.plot(gain_real[k], label=f"user {k} ave gain")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'gain simulation {pos_idx}')
#     plt.show()
#
#     plt.figure()
#     plt.plot(interference_estimate[0], label='user 0 interference estimate', linestyle='--')
#     plt.plot(interference_estimate[1], label='user 1 interference estimate', linestyle='--')
#     for k in range(K):
#         plt.plot(interference_real[k], label=f"user {k} ave interference")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'interference simulation {pos_idx}')
#     plt.show()


# # 测试用户多天线EZF场景性能
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
# def get_largescale(dub, fc):
#
#     beta_dB = 31.84 + 21.5 * np.log10(dub) + 19 * np.log10(fc * 1e-3)
#
#     return 10 ** (- beta_dB/10)
#
#
# K = 2
# Mt = 256
# Mr = 4
# n_layer = 2     # 每个用户双流传输
# n_stream = K * n_layer
# kai = np.pi
# delta_theta = 5/180 * np.pi
# BS_pos = [300, 300]
# u_pos = [[[ 52, 201 ], [ 178, 262 ]], [[ 52, 201 ], [ 178, 62 ]], [[ 52, 201 ], [ 178, 562 ]]]
# h_U = 1.65
# h_BS = 15
# fc = 3500
# noise = 10 ** (-142/10 - 3) * 30 * 1e3
# P0 = 10 ** (10/10 - 3)
#
# T = 160
# N = 50
# I = 0.013
# rho = 0.9966
#
# for pos_idx in range(3):
#     # 生成信道空间相关矩阵
#     Rt = np.zeros((K, Mt, Mt), dtype=np.complex128)
#     for k in range(K):
#         for m in range(Mt):
#             for n in range(Mt):
#                 theta_bar = calculate_AOD(BS_pos[0], BS_pos[1], u_pos[pos_idx][k][0], u_pos[pos_idx][k][1])
#                 Rt[k][m][n] = 1 / Mt * np.exp(
#                     -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                     kai * (m - n) * delta_theta * np.sin(theta_bar))
#
#     Rr = np.zeros((K, Mr, Mr), dtype=np.complex128)
#     for k in range(K):
#         for m in range(Mr):
#             for n in range(Mr):
#                 theta_bar = calculate_AOD(u_pos[pos_idx][k][0], u_pos[pos_idx][k][1], BS_pos[0], BS_pos[1])
#                 Rr[k][m][n] = 1 / Mr * np.exp(
#                     -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                     kai * (m - n) * delta_theta * np.sin(theta_bar))
#
#     Hl = np.zeros((K, Mr, Mt))
#     for k in range(K):
#         dub = get_dis(u_pos[pos_idx][k], BS_pos, h_U, h_BS)
#         Hl[k, :, :] = np.sqrt(get_largescale(dub, fc))
#     Hs, _, combiner_equal_gain = generate_gaussian_channel(Rr, Rt, n_layer)
#     H = Hl * Hs
#
#     # 用基本的一套信道生成一次EZF预编码，并记录功率
#     H_equal = np.zeros((n_stream, Mt), dtype=np.complex128)
#     for k in range(K):
#         U, s, VT = np.linalg.svd(H[k, :, :])
#         H_equal[k * n_layer : (k+1) * n_layer, :] = U[:, :n_layer].conj().T @ H[k]
#     inv = np.linalg.inv(H_equal.conj() @ H_equal.T + n_stream * noise / P0 * np.eye(n_stream))
#     precoder = H_equal.T @ inv
#     precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#     P = []
#     for j in range(n_stream):
#         P.append(np.linalg.norm(precoder[:, j]) ** 2)
#
#     gain = []
#     interference = []
#
#     for k in range(K):
#         for l in range(n_layer):
#             gain.append((Hl[k, 0, 0] ** 2) * P[k * n_layer + l] * (combiner_equal_gain[k, l] ** 2) * np.trace(Rt[k, :, :] @ Rt[k, :, :]).real)
#             interference_temp_sum = 0
#             for j in range(n_stream):
#                 if j == k * n_layer + l:
#                     continue
#                 k0 = j // n_layer
#                 l0 = j % n_layer
#                 interference_temp_sum += ((Hl[k, 0, 0] ** 2) * P[j] * (combiner_equal_gain[k, l] ** 2)
#                                       * np.trace(Rt[k0, :, :] @ Rt[k, :, :]).real)
#             interference.append(interference_temp_sum)
#
#     sinr_estimate = [[] for _ in range(n_stream)]
#     gain_estimate = [[] for _ in range(n_stream)]
#     interference_estimate = [[] for _ in range(n_stream)]
#     for t in range(T):
#         for j in range(n_stream):
#             sinr_estimate[j].append(10 * np.log10((gain[j] * (rho ** (2 * t))) / (interference[j] * (1 - rho ** (2 * t)) + noise)))
#             gain_estimate[j].append(10 * np.log10(gain[j] * (rho ** (2 * t))))
#             interference_estimate[j].append(10 * np.log10(interference[j] * (1 - rho ** (2 * t)) + 1e-20))
#
#     # 蒙特卡洛仿真实际信道老化情况
#     gain_real = [[0] * T for _ in range(n_stream)]
#     interference_real = [[0] * T for _ in range(n_stream)]
#     sinr_real = [[0] * T for _ in range(n_stream)]
#     for n in range(N):
#         print(f"n = {n}")
#         Hs_m, _, _ = generate_gaussian_channel(Rr, Rt, n_layer)
#         H_m = Hl * Hs_m
#         H_equal_m = np.zeros((n_stream, Mt), dtype=np.complex128)
#         combiner = np.zeros((n_stream, Mr), dtype=np.complex128)
#         for k in range(K):
#             U, s, VT = np.linalg.svd(H_m[k, :, :])
#             combiner[k * n_layer: (k + 1) * n_layer, :] = U[:, :n_layer].T
#             H_equal_m[k * n_layer: (k + 1) * n_layer, :] = combiner[k * n_layer: (k + 1) * n_layer, :].conj() @ H_m[k]
#         inv = np.linalg.inv(H_equal_m.conj() @ H_equal_m.T + n_stream * noise / P0 * np.eye(n_stream))
#         precoder = H_equal_m.T @ inv
#         precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#         gain_temp = [0] * n_stream
#         interference_temp = [0] * n_stream
#         for t in range(T):
#             for k in range(K):
#                 for l in range(n_layer):
#                     gain_temp[k * n_layer + l] = np.linalg.norm(combiner[k * n_layer + l, :] @ H_m[k, :, :].conj() @ precoder[:, k * n_layer + l]) ** 2
#                     gain_real[k * n_layer + l][t] += 10 * np.log10(gain_temp[k * n_layer + l])
#                     interference_temp_sum = 0
#                     for j in range(n_stream):
#                         if j == k * n_layer + l:
#                             continue
#                         interference_temp_sum += np.linalg.norm(combiner[k * n_layer + l, :] @ H_m[k, :, :].conj() @ precoder[:, j]) ** 2
#                     interference_temp[k * n_layer + l] = interference_temp_sum
#                     interference_real[k * n_layer + l][t] += 10 * np.log10(interference_temp[k * n_layer + l])
#                     sinr_real[k * n_layer + l][t] += 10 * np.log10(gain_temp[k * n_layer + l] / (interference_temp[k * n_layer + l] + noise))
#             G , _, _ = generate_gaussian_channel(Rr, Rt, n_layer)
#             Hs_m = rho * Hs_m + np.sqrt(1 - rho ** 2) * G
#             H_m = Hl * Hs_m
#     for t in range(T):
#         for k in range(n_stream):
#             gain_real[k][t] /= N
#             interference_real[k][t] /= N
#             sinr_real[k][t] /= N
#
#     plt.figure()
#     for j in range(n_stream):
#         plt.plot(sinr_estimate[j], label=f'stream {j} sinr estimate', linestyle='--')
#         plt.plot(sinr_real[j], label=f"stream {j} ave sinr")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'sinr simulation {pos_idx}')
#     plt.show()
#
#     plt.figure()
#     for j in range(n_stream):
#         plt.plot(gain_estimate[j], label=f'stream {j} gain estimate', linestyle='--')
#         plt.plot(gain_real[j], label=f"stream {j} ave gain")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'gain simulation {pos_idx}')
#     plt.show()
#
#     plt.figure()
#     for j in range(n_stream):
#         plt.plot(interference_estimate[j], label=f'stream {j} interference estimate', linestyle='--')
#         plt.plot(interference_real[j], label=f"stream {j} ave interference")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'interference simulation {pos_idx}')
#     plt.show()

#
# # 测试用户多天线WMMSE场景性能
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
# def get_largescale(dub, fc):
#
#     beta_dB = 31.84 + 21.5 * np.log10(dub) + 19 * np.log10(fc * 1e-3)
#
#     return 10 ** (- beta_dB/10)
#
#
# K = 2
# Mt = 256
# Mr = 4
# n_layer = 2     # 每个用户传输流数
# n_stream = K * n_layer
# WMMSE_max_iteration = 5     # WMMSE最大迭代计算次数
# kai = np.pi
# delta_theta = 5/180 * np.pi
# BS_pos = [300, 300]
# u_pos = [[[ 52, 201 ], [ 178, 262 ]], [[ 52, 201 ], [ 178, 62 ]], [[ 52, 201 ], [ 178, 562 ]]]
# h_U = 1.65
# h_BS = 15
# fc = 3500
# noise = 10 ** (-174/10 - 3) * 30 * 1e3
# P0 = 10 ** (23.18/10 - 3)
#
# T = 160
# N = 5
# I = 0.013
# rho = 0.9966
#
# for pos_idx in range(3):
#     # 生成信道空间相关矩阵
#     Rt = np.zeros((K, Mt, Mt), dtype=np.complex128)
#     for k in range(K):
#         for m in range(Mt):
#             for n in range(Mt):
#                 theta_bar = calculate_AOD(BS_pos[0], BS_pos[1], u_pos[pos_idx][k][0], u_pos[pos_idx][k][1])
#                 Rt[k][m][n] = 1 / Mt * np.exp(
#                     -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                     kai * (m - n) * delta_theta * np.sin(theta_bar))
#
#     Rr = np.zeros((K, Mr, Mr), dtype=np.complex128)
#     for k in range(K):
#         for m in range(Mr):
#             for n in range(Mr):
#                 theta_bar = calculate_AOD(u_pos[pos_idx][k][0], u_pos[pos_idx][k][1], BS_pos[0], BS_pos[1])
#                 Rr[k][m][n] = 1 / Mr * np.exp(
#                     -1j * kai * (m - n) * np.cos(theta_bar)) * np.sinc(
#                     kai * (m - n) * delta_theta * np.sin(theta_bar))
#
#     Hl = np.zeros((K, Mr, Mt))
#     for k in range(K):
#         dub = get_dis(u_pos[pos_idx][k], BS_pos, h_U, h_BS)
#         Hl[k, :, :] = np.sqrt(get_largescale(dub, fc))
#     Hs, _, combiner_equal_gain = generate_gaussian_channel(Rr, Rt, n_layer)
#     H = Hl * Hs
#
#     # 用基本的一套信道生成一次WMMSE预编码，并记录功率
#     # 先以EZF预编码初始化
#     H_equal = np.zeros((n_stream, Mt), dtype=np.complex128)
#     combiner = np.zeros((n_stream, Mr), dtype=np.complex128)
#     for k in range(K):
#         U, s, VT = np.linalg.svd(H[k, :, :])
#         combiner[k * n_layer: (k + 1) * n_layer, :] = U[:, :n_layer].T
#         H_equal[k * n_layer: (k + 1) * n_layer, :] = combiner[k * n_layer: (k + 1) * n_layer, :].conj() @ H[k]
#     inv = np.linalg.inv(H_equal.conj() @ H_equal.T + n_stream * noise / P0 * np.eye(n_stream))
#     precoder = H_equal.T @ inv
#     precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#
#     # # WMMSE迭代
#     # combiner = np.zeros((Mr, n_stream), dtype=np.complex128)
#     # for iteration in range(WMMSE_max_iteration):
#     #     receivers = {}
#     #     mse_weights = {}
#     #
#     #     # 更新MMSE接收机
#     #     for k in range(K):
#     #
#     #         H_user = H[k, :, :].T
#     #
#     #         W_user = precoder[:, k * n_layer: (k + 1) * n_layer]
#     #
#     #         # 接收信号协方差
#     #         R_yy = noise * np.eye(Mr, dtype=complex)
#     #         for j in range(K):
#     #             if j != k:
#     #                 W_other = precoder[:, j * n_layer: (j + 1) * n_layer]
#     #                 R_yy += H_user.conj().T @ W_other @ W_other.conj().T @ H_user
#     #
#     #         # MMSE接收机
#     #         try:
#     #             signal = H_user.conj().T @ W_user
#     #             G_mmse = np.linalg.inv(R_yy + 1e-10 * np.eye(Mr)) @ signal
#     #             receivers[k] = G_mmse
#     #             combiner[:, k * n_layer: (k + 1) * n_layer] = G_mmse
#     #
#     #             # MSE权重
#     #             I = np.eye(n_layer, dtype=complex)
#     #             MSE_k = I - G_mmse.conj().T @ signal - signal.conj().T @ G_mmse + \
#     #                     G_mmse.conj().T @ R_yy @ G_mmse
#     #             mse_weights[k] = np.linalg.inv(MSE_k + 1e-10 * I)
#     #         except:
#     #             pass
#     #
#     #     # 更新预编码器
#     #     for k in range(K):
#     #
#     #         H_user = H[k, :, :].T
#     #
#     #         # 构建干扰矩阵
#     #         A = 1e-10 * np.eye(Mt, dtype=complex)
#     #         for j in range(K):
#     #             if j != k:
#     #                 H_j = H[j, :, :].T
#     #                 G_j = receivers[j]
#     #                 U_j = mse_weights[j]
#     #                 A += H_j @ G_j @ U_j @ G_j.conj().T @ H_j.conj().T
#     #
#     #         # 更新预编码
#     #         try:
#     #             B = H_user @ receivers[k] @ mse_weights[k]
#     #             W_new = np.linalg.inv(A) @ B
#     #
#     #             # 功率约束
#     #             power = np.trace(W_new @ W_new.conj().T).real
#     #             if power > 0:
#     #                 n_scheduled = K
#     #                 power_budget = P0 / max(n_scheduled, 1)
#     #                 W_new = W_new * np.sqrt(power_budget / power)
#     #
#     #             precoder[:, k * n_layer: (k + 1) * n_layer] = W_new
#     #         except:
#     #             pass
#     #
#     #     # 功率归一化
#     #     precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#     #
#     # # combiner归一化
#     # for s in range(n_stream):
#     #     combiner[:, s] = combiner[:, s] / np.linalg.norm(combiner[:, s])
#     #
#     # # 我仿真时用的combiner还需要共轭转置才适配
#     # combiner = combiner.conj().T
#
#     P = []
#     for j in range(n_stream):
#         P.append(np.linalg.norm(precoder[:, j]) ** 2)
#
#     gain = []
#     interference = []
#
#     for k in range(K):
#         for l in range(n_layer):
#             gain.append((Hl[k, 0, 0] ** 2) * P[k * n_layer + l] * (combiner_equal_gain[k, l] ** 2))
#             interference_temp_sum = 0
#             for j in range(n_stream):
#                 if j == k * n_layer + l:
#                     continue
#                 k0 = j // n_layer
#                 l0 = j % n_layer
#                 interference_temp_sum += ((Hl[k, 0, 0] ** 2) * P[j] * (combiner_equal_gain[k, l] ** 2)
#                                       * np.trace(Rt[k0, :, :] @ Rt[k, :, :]).real)
#             interference.append(interference_temp_sum)
#
#     sinr_estimate = [[] for _ in range(n_stream)]
#     gain_estimate = [[] for _ in range(n_stream)]
#     interference_estimate = [[] for _ in range(n_stream)]
#     for t in range(T):
#         for j in range(n_stream):
#             sinr_estimate[j].append(10 * np.log10((gain[j] * (rho ** (2 * t))) / (interference[j] * (1 - rho ** (2 * t)) + noise)))
#             gain_estimate[j].append(10 * np.log10(gain[j] * (rho ** (2 * t))))
#             interference_estimate[j].append(10 * np.log10(interference[j] * (1 - rho ** (2 * t)) + 1e-20))
#
#     # 蒙特卡洛仿真实际信道老化情况
#     gain_real = [[0] * T for _ in range(n_stream)]
#     interference_real = [[0] * T for _ in range(n_stream)]
#     sinr_real = [[0] * T for _ in range(n_stream)]
#     for n in range(N):
#         print(f"n = {n}")
#         Hs_m, _, _ = generate_gaussian_channel(Rr, Rt, n_layer)
#         H_m = Hl * Hs_m
#         H_equal_m = np.zeros((n_stream, Mt), dtype=np.complex128)
#         combiner = np.zeros((n_stream, Mr), dtype=np.complex128)
#
#         for k in range(K):
#             U, s, VT = np.linalg.svd(H_m[k, :, :])
#             combiner[k * n_layer: (k + 1) * n_layer, :] = U[:, :n_layer].T
#             H_equal_m[k * n_layer: (k + 1) * n_layer, :] = combiner[k * n_layer: (k + 1) * n_layer, :].conj() @ H_m[k]
#         inv = np.linalg.inv(H_equal_m.conj() @ H_equal_m.T + n_stream * noise / P0 * np.eye(n_stream))
#         precoder = H_equal_m.T @ inv
#         precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#
#         # # WMMSE迭代
#         # combiner = np.zeros((Mr, n_stream), dtype=np.complex128)
#         # for iteration in range(WMMSE_max_iteration):
#         #     receivers = {}
#         #     mse_weights = {}
#         #
#         #     # 更新MMSE接收机
#         #     for k in range(K):
#         #
#         #         H_user = H_m[k, :, :].T
#         #
#         #         W_user = precoder[:, k * n_layer: (k + 1) * n_layer]
#         #
#         #         # 接收信号协方差
#         #         R_yy = noise * np.eye(Mr, dtype=complex)
#         #         for j in range(K):
#         #             if j != k:
#         #                 W_other = precoder[:, j * n_layer: (j + 1) * n_layer]
#         #                 R_yy += H_user.conj().T @ W_other @ W_other.conj().T @ H_user
#         #
#         #         # MMSE接收机
#         #         try:
#         #             signal = H_user.conj().T @ W_user
#         #             G_mmse = np.linalg.inv(R_yy + 1e-10 * np.eye(Mr)) @ signal
#         #             receivers[k] = G_mmse
#         #             combiner[:, k * n_layer: (k + 1) * n_layer] = G_mmse
#         #
#         #             # MSE权重
#         #             I = np.eye(n_layer, dtype=complex)
#         #             MSE_k = I - G_mmse.conj().T @ signal - signal.conj().T @ G_mmse + \
#         #                     G_mmse.conj().T @ R_yy @ G_mmse
#         #             mse_weights[k] = np.linalg.inv(MSE_k + 1e-10 * I)
#         #         except:
#         #             pass
#         #
#         #     # 更新预编码器
#         #     for k in range(K):
#         #
#         #         H_user = H_m[k, :, :].T
#         #
#         #         # 构建干扰矩阵
#         #         A = 1e-10 * np.eye(Mt, dtype=complex)
#         #         for j in range(K):
#         #             if j != k:
#         #                 H_j = H_m[j, :, :].T
#         #                 G_j = receivers[j]
#         #                 U_j = mse_weights[j]
#         #                 A += H_j @ G_j @ U_j @ G_j.conj().T @ H_j.conj().T
#         #
#         #         # 更新预编码
#         #         try:
#         #             B = H_user @ receivers[k] @ mse_weights[k]
#         #             W_new = np.linalg.inv(A) @ B
#         #
#         #             # 功率约束
#         #             power = np.trace(W_new @ W_new.conj().T).real
#         #             if power > 0:
#         #                 n_scheduled = K
#         #                 power_budget = P0 / max(n_scheduled, 1)
#         #                 W_new = W_new * np.sqrt(power_budget / power)
#         #
#         #             precoder[:, k * n_layer: (k + 1) * n_layer] = W_new
#         #         except:
#         #             pass
#         #
#         #     # 功率归一化
#         #     precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder
#         #
#         # # combiner归一化
#         # for s in range(n_stream):
#         #     combiner[:, s] = combiner[:, s] / np.linalg.norm(combiner[:, s])
#         #
#         # # 我仿真时用的combiner还需要共轭转置才适配
#         # combiner = combiner.conj().T
#
#         gain_temp = [0] * n_stream
#         interference_temp = [0] * n_stream
#         for t in range(T):
#             for k in range(K):
#                 for l in range(n_layer):
#                     gain_temp[k * n_layer + l] = np.linalg.norm(combiner[k * n_layer + l, :] @ H_m[k, :, :].conj() @ precoder[:, k * n_layer + l]) ** 2
#                     gain_real[k * n_layer + l][t] += 10 * np.log10(gain_temp[k * n_layer + l])
#                     interference_temp_sum = 0
#                     for j in range(n_stream):
#                         if j == k * n_layer + l:
#                             continue
#                         interference_temp_sum += np.linalg.norm(combiner[k * n_layer + l, :] @ H_m[k, :, :].conj() @ precoder[:, j]) ** 2
#                     interference_temp[k * n_layer + l] = interference_temp_sum
#                     interference_real[k * n_layer + l][t] += 10 * np.log10(interference_temp[k * n_layer + l])
#                     sinr_real[k * n_layer + l][t] += 10 * np.log10(gain_temp[k * n_layer + l] / (interference_temp[k * n_layer + l] + noise))
#             G , _, _ = generate_gaussian_channel(Rr, Rt, n_layer)
#             Hs_m = rho * Hs_m + np.sqrt(1 - rho ** 2) * G
#             H_m = Hl * Hs_m
#     for t in range(T):
#         for k in range(n_stream):
#             gain_real[k][t] /= N
#             interference_real[k][t] /= N
#             sinr_real[k][t] /= N
#
#     plt.figure()
#     for j in range(n_stream):
#         plt.plot(sinr_estimate[j], label=f'stream {j} sinr estimate', linestyle='--')
#         plt.plot(sinr_real[j], label=f"stream {j} ave sinr")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'sinr simulation {pos_idx}')
#     plt.show()
#
#     plt.figure()
#     for j in range(n_stream):
#         plt.plot(gain_estimate[j], label=f'stream {j} gain estimate', linestyle='--')
#         plt.plot(gain_real[j], label=f"stream {j} ave gain")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'gain simulation {pos_idx}')
#     plt.show()
#
#     plt.figure()
#     for j in range(n_stream):
#         plt.plot(interference_estimate[j], label=f'stream {j} interference estimate', linestyle='--')
#         plt.plot(interference_real[j], label=f"stream {j} ave interference")
#     plt.grid(True)
#     plt.legend()
#     plt.title(f'interference simulation {pos_idx}')
#     plt.show()


# 测试quadriga信道下用户多天线WMMSE场景性能
from scipy.io import loadmat
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
# fc = 3500
noise = 10 ** (-174/10 - 3) * 30 * 1e3 * 12
P0 = 10 ** (23/10 - 3)

T = 160
N = 1
I = 0.013
rho = [0.9966, 0.9966]
start_TTI = 160

quadriga_dir = "/home/fj24/25_8_Huawei_multiTTI/信道/quadriga_channel_test"

# 读取quadriga信道
qg_src = QuadrigaChannelSource(quadriga_dir)
H = {}
Hl = {}
Rt = {}
Rr = {}
for ue_id in range(K):
    H[ue_id] = qg_src.get_H(ue_id, 0 + start_TTI-T)       # [NRB, NBS, N_RX]
    # assert H[ue_id].shape == (N_RB, Mt, Mr), "channel shape mismatch!"

    Hl[ue_id], Hs = separate_large_scale_fading(H[ue_id])
    Rr[ue_id], Rt[ue_id] = estimate_corr_matrix(Hs)

    H_next = qg_src.get_H(ue_id, 0 + start_TTI)
    rho_gap = np.linalg.norm(np.trace(H[ue_id][0] @ H_next[0].conj().T) / np.linalg.norm(H[ue_id][0], 'fro') ** 2)
    rho[ue_id] = rho_gap ** (1 / T)
# rho[1] = 0.9965
# test1 = np.trace(Rt[0] @ Rt[1]).real
# test2 = np.trace(Rt[0]).real
# test3 = np.trace(Rt[1]).real
# 用基本的一套信道生成一次WMMSE预编码，并记录功率
# 先以EZF预编码初始化
H_equal = np.zeros((n_stream, Mt), dtype=np.complex128)
combiner = np.zeros((n_stream, Mr), dtype=np.complex128)
for k in range(K):
    U, s, VT = np.linalg.svd(H[k][0].conj().T)
    combiner[k * n_layer: (k + 1) * n_layer, :] = U[:, :n_layer].conj().T
    H_equal[k * n_layer: (k + 1) * n_layer, :] = combiner[k * n_layer: (k + 1) * n_layer, :] @ H[k][0].conj().T
inv = np.linalg.inv(H_equal @ H_equal.conj().T + n_stream * noise / P0 * np.eye(n_stream))
precoder = H_equal.conj().T @ inv
precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder

# WMMSE迭代
combiner = np.zeros((Mr, n_stream), dtype=np.complex128)
for iteration in range(WMMSE_max_iteration):
    receivers = {}
    mse_weights = {}

    # 更新MMSE接收机
    for k in range(K):

        H_user = H[k][0]

        W_user = precoder[:, k * n_layer: (k + 1) * n_layer]

        # 接收信号协方差
        R_yy = noise * np.eye(Mr, dtype=complex)
        for j in range(K):
            if j != k:
                W_other = precoder[:, j * n_layer: (j + 1) * n_layer]
                R_yy += H_user.conj().T @ W_other @ W_other.conj().T @ H_user

        # MMSE接收机
        try:
            signal = H_user.conj().T @ W_user
            G_mmse = np.linalg.inv(R_yy + 1e-10 * np.eye(Mr)) @ signal
            receivers[k] = G_mmse
            combiner[:, k * n_layer: (k + 1) * n_layer] = G_mmse

            # MSE权重
            I = np.eye(n_layer, dtype=complex)
            MSE_k = I - G_mmse.conj().T @ signal - signal.conj().T @ G_mmse + \
                    G_mmse.conj().T @ R_yy @ G_mmse
            mse_weights[k] = np.linalg.inv(MSE_k + 1e-10 * I)
        except:
            pass

    # 更新预编码器
    for k in range(K):

        H_user = H[k][0]

        # 构建干扰矩阵
        A = 1e-10 * np.eye(Mt, dtype=complex)
        for j in range(K):
            if j != k:
                H_j = H[j][0]
                G_j = receivers[j]
                U_j = mse_weights[j]
                A += H_j @ G_j @ U_j @ G_j.conj().T @ H_j.conj().T

        # 更新预编码
        try:
            B = H_user @ receivers[k] @ mse_weights[k]
            W_new = np.linalg.inv(A) @ B

            # 功率约束
            power = np.trace(W_new @ W_new.conj().T).real
            if power > 0:
                n_scheduled = K
                power_budget = P0 / max(n_scheduled, 1)
                W_new = W_new * np.sqrt(power_budget / power)

            precoder[:, k * n_layer: (k + 1) * n_layer] = W_new
        except:
            pass

    # 功率归一化
    precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder

# combiner归一化
for s in range(n_stream):
    combiner[:, s] = combiner[:, s] / np.linalg.norm(combiner[:, s])

# 我仿真时用的combiner还需要共轭转置才适配
combiner = combiner.conj().T

P = []
for j in range(n_stream):
    P.append(np.linalg.norm(precoder[:, j]) ** 2)

gain = []
gain2 = []
interference = []
self_interference = []

for k in range(K):
    eigenvalues_r, _ = np.linalg.eigh(Rr[k])
    eigenvalues_r = eigenvalues_r[::-1]
    lambda_sqrt = np.sqrt(eigenvalues_r)
    combiner_equal_gain = lambda_sqrt[:n_layer]

    temp1 = np.trace(Rt[k] @ Rt[k]).real / Mt

    # eigenvalues_t, _ = np.linalg.eigh(Rt[k])
    # eigenvalues_t = eigenvalues_t[::-1]
    # temp2 = sum([e ** 2 for e in eigenvalues_t]) / Mt
    # temp3 = sum(eigenvalues_t)
    # temp4 = sum([e ** 2 for e in eigenvalues_t])
    # temp5 = eigenvalues_t[0] ** 2

    for l in range(n_layer):
        gain.append((Hl[k] ** 2) * P[k * n_layer + l] * (combiner_equal_gain[l] ** 2) * Mt)
        # gain2.append((Hl[k] ** 2) * P[k * n_layer + l] * (combiner_equal_gain[l] ** 2)* np.trace(Rt[k] @ Rt[k]).real / Mt)
        # gain.append(Hl[k] * P[k * n_layer + l] * (combiner_equal_gain[l] ** 2))
        interference_temp_sum = 0
        for j in range(n_stream):
            if j == k * n_layer + l:
                continue
            k0 = j // n_layer
            l0 = j % n_layer
            if k0 == k:
                _, s, _ = np.linalg.svd(Rt[k])
                self_interference.append(((Hl[k] ** 2) * P[j] * (combiner_equal_gain[l] ** 2)))
                interference_temp_sum += ((Hl[k] ** 2) * P[j] * (combiner_equal_gain[l] ** 2))
            else:
                interference_temp_sum += ((Hl[k] ** 2) * P[j] * (combiner_equal_gain[l] ** 2)
                                      * np.trace(Rt[k0] @ Rt[k]).real / Mt)
            # interference_temp_sum += (Hl[k] * P[j] * (combiner_equal_gain[l] ** 2)
            #                           * np.trace(Rt[k0] @ Rt[k]).real)
        interference.append(interference_temp_sum)

sinr_estimate = [[] for _ in range(n_stream)]
gain_estimate = [[] for _ in range(n_stream)]
interference_estimate = [[] for _ in range(n_stream)]
self_interference_estimate = [[] for _ in range(n_stream)]
for t in range(T):
    for j in range(n_stream):
        sinr_estimate[j].append(10 * np.log10((gain[j] * (rho[j // n_layer] ** (2 * t))) / (interference[j] * (1 - rho[j // n_layer] ** (2 * t)) + noise)))
        # sinr_estimate[j].append(
        #     10 * np.log10((gain[j] * (rho[j] ** (2 * t)) + gain2[j] * (1 - (rho[j] ** (2 * t)))) / (interference[j] * (1 - rho[j] ** (2 * t)) + noise)))
        gain_estimate[j].append(10 * np.log10(gain[j] * (rho[j // n_layer] ** (2 * t))))
        # gain_estimate[j].append(10 * np.log10(gain[j] * (rho[j] ** (2 * t)) + gain2[j] * (1 - (rho[j] ** (2 * t)))))
        interference_estimate[j].append(10 * np.log10(interference[j] * (1 - rho[j // n_layer] ** (2 * t))))
        # self_interference_estimate[j].append(10 * np.log10(self_interference[j] * (1 - rho[j // n_layer] ** (2 * t))))

# 蒙特卡洛仿真实际信道老化情况
gain_real = [[0] * T for _ in range(n_stream)]
interference_real = [[0] * T for _ in range(n_stream)]
self_interference_real = [[0] * T for _ in range(n_stream)]
sinr_real = [[0] * T for _ in range(n_stream)]
for n in range(N):
    print(f"n = {n}")

    for ue_id in range(K):
        H[ue_id] = qg_src.get_H(ue_id, 0 + start_TTI)  # [NRB, NBS, N_RX]
        # assert H[ue_id].shape == (N_RB, Mt, Mr), "channel shape mismatch!"

    H_equal = np.zeros((n_stream, Mt), dtype=np.complex128)
    combiner = np.zeros((n_stream, Mr), dtype=np.complex128)
    for k in range(K):
        U, s, VT = np.linalg.svd(H[k][n].conj().T)
        combiner[k * n_layer: (k + 1) * n_layer, :] = U[:, :n_layer].conj().T
        H_equal[k * n_layer: (k + 1) * n_layer, :] = combiner[k * n_layer: (k + 1) * n_layer, :] @ H[k][n].conj().T
    inv = np.linalg.inv(H_equal @ H_equal.conj().T + n_stream * noise / P0 * np.eye(n_stream))
    precoder = H_equal.conj().T @ inv
    precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder

    # WMMSE迭代
    combiner = np.zeros((Mr, n_stream), dtype=np.complex128)
    for iteration in range(WMMSE_max_iteration):
        receivers = {}
        mse_weights = {}

        # 更新MMSE接收机
        for k in range(K):

            H_user = H[k][n]

            W_user = precoder[:, k * n_layer: (k + 1) * n_layer]

            # 接收信号协方差
            R_yy = noise * np.eye(Mr, dtype=complex)
            for j in range(K):
                if j != k:
                    W_other = precoder[:, j * n_layer: (j + 1) * n_layer]
                    R_yy += H_user.conj().T @ W_other @ W_other.conj().T @ H_user

            # MMSE接收机
            try:
                signal = H_user.conj().T @ W_user
                G_mmse = np.linalg.inv(R_yy + 1e-10 * np.eye(Mr)) @ signal
                receivers[k] = G_mmse
                combiner[:, k * n_layer: (k + 1) * n_layer] = G_mmse

                # MSE权重
                I = np.eye(n_layer, dtype=complex)
                MSE_k = I - G_mmse.conj().T @ signal - signal.conj().T @ G_mmse + \
                        G_mmse.conj().T @ R_yy @ G_mmse
                mse_weights[k] = np.linalg.inv(MSE_k + 1e-10 * I)
            except:
                pass

        # 更新预编码器
        for k in range(K):

            H_user = H[k][n]

            # 构建干扰矩阵
            A = 1e-10 * np.eye(Mt, dtype=complex)
            for j in range(K):
                if j != k:
                    H_j = H[j][n]
                    G_j = receivers[j]
                    U_j = mse_weights[j]
                    A += H_j @ G_j @ U_j @ G_j.conj().T @ H_j.conj().T

            # 更新预编码
            try:
                B = H_user @ receivers[k] @ mse_weights[k]
                W_new = np.linalg.inv(A) @ B

                # 功率约束
                power = np.trace(W_new @ W_new.conj().T).real
                if power > 0:
                    n_scheduled = K
                    power_budget = P0 / max(n_scheduled, 1)
                    W_new = W_new * np.sqrt(power_budget / power)

                precoder[:, k * n_layer: (k + 1) * n_layer] = W_new
            except:
                pass

        # 功率归一化
        precoder = np.sqrt(P0 / np.trace(precoder.conj().T @ precoder)) * precoder

    # combiner归一化
    for s in range(n_stream):
        combiner[:, s] = combiner[:, s] / np.linalg.norm(combiner[:, s])

    # 我仿真时用的combiner还需要共轭转置才适配
    combiner = combiner.conj().T

    gain_temp = [0] * n_stream
    interference_temp = [0] * n_stream
    self_interference_temp = [0] * n_stream
    for t in range(T):

        for k in range(K):

            for l in range(n_layer):
                gain_temp[k * n_layer + l] = np.linalg.norm(combiner[k * n_layer + l, :] @ H[k][n].conj().T @ precoder[:, k * n_layer + l]) ** 2
                # gain_real[k * n_layer + l][t] += 10 * np.log10(gain_temp[k * n_layer + l])
                gain_real[k * n_layer + l][t] += gain_temp[k * n_layer + l]
                interference_temp_sum = 0
                for j in range(n_stream):
                    if j == k * n_layer + l:
                        continue
                    if j // n_layer == k:
                        self_interference_temp[k * n_layer + l] = np.linalg.norm(combiner[k * n_layer + l, :] @ H[k][n].conj().T @ precoder[:, j]) ** 2
                    interference_temp_sum += np.linalg.norm(combiner[k * n_layer + l, :] @ H[k][n].conj().T @ precoder[:, j]) ** 2
                interference_temp[k * n_layer + l] = interference_temp_sum
                # interference_real[k * n_layer + l][t] += 10 * np.log10(interference_temp[k * n_layer + l])
                interference_real[k * n_layer + l][t] += interference_temp[k * n_layer + l]
                # self_interference_real[k * n_layer + l][t] += self_interference_temp[k * n_layer + l]
                # sinr_real[k * n_layer + l][t] += 10 * np.log10(gain_temp[k * n_layer + l] / (interference_temp[k * n_layer + l] + noise))
                sinr_real[k * n_layer + l][t] += gain_temp[k * n_layer + l] / (interference_temp[k * n_layer + l] + noise)

        for ue_id in range(K):
            H[ue_id] = qg_src.get_H(ue_id, t+1 + start_TTI)  # [NRB, NBS, N_RX]
            # assert H[ue_id].shape == (N_RB, Mt, Mr), "channel shape mismatch!"

for t in range(T):
    for k in range(n_stream):
        gain_real[k][t] /= N
        interference_real[k][t] /= N
        self_interference_real[k][t] /= N
        sinr_real[k][t] /= N
        gain_real[k][t] = 10 * np.log10(gain_real[k][t])
        interference_real[k][t] = 10 * np.log10(interference_real[k][t])
        self_interference_real[k][t] = 10 * np.log10(self_interference_real[k][t])
        sinr_real[k][t] = 10 * np.log10(sinr_real[k][t])

plt.figure()
for j in range(n_stream):
    plt.plot(sinr_estimate[j], label=f'stream {j} sinr estimate', linestyle='--')
    plt.plot(sinr_real[j], label=f"stream {j} ave sinr")
plt.grid(True)
plt.legend()
plt.title(f'sinr simulation from TTI = {start_TTI}')
plt.show()

plt.figure()
for j in range(n_stream):
    plt.plot(gain_estimate[j], label=f'stream {j} gain estimate', linestyle='--')
    plt.plot(gain_real[j], label=f"stream {j} ave gain")
plt.grid(True)
plt.legend()
plt.title(f'gain simulation from TTI = {start_TTI}')
plt.show()

plt.figure()
for j in range(n_stream):
    plt.plot(interference_estimate[j], label=f'stream {j} interference estimate', linestyle='--')
    plt.plot(interference_real[j], label=f"stream {j} ave interference")
plt.grid(True)
plt.legend()
plt.title(f'interference simulation from TTI = {start_TTI}')
plt.show()

# plt.figure()
# for j in range(n_stream):
#     plt.plot(self_interference_estimate[j], label=f'stream {j} self interference estimate', linestyle='--')
#     plt.plot(self_interference_real[j], label=f"stream {j} ave self interference")
# plt.grid(True)
# plt.legend()
# plt.title(f'self interference simulation from TTI = {start_TTI}')
# plt.show()