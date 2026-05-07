import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_channel(R, total_samples):
    """
    高效批量生成复高斯随机向量

    Args:
        R: 协方差矩阵 (M, M)

    Returns:
        形状为 (batch_size, num_vectors, M) 的数组
    """
    M = R.shape[0]
    L = np.linalg.cholesky(R)

    # 一次性生成所有随机数
    h_real = np.random.randn(total_samples, M)
    h_imag = np.random.randn(total_samples, M)
    h = (h_real + 1j * h_imag) / np.sqrt(2)

    # 批量变换
    h = (L @ h.T).T

    # 重塑为批次格式
    return h.reshape(total_samples, M)


M = 32          # 天线数
r = 0.2 * np.exp(-1j * np.pi/4)         # 信道空间相关系数
T = 1000000       # 蒙特卡洛仿真次数

# 生成信道空间相关矩阵
R = np.zeros((M, M), dtype=np.complex128)
for i in range(M):
    for j in range(M):
        R[i][j] = r ** np.abs(i - j) if i <= j else np.conj(r ** np.abs(i - j))

# 生成DFT码本，等效于在0到Π上均匀分布的角度
# temp = np.outer(np.arange(M), np.arange(M))
DFT_codebook = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)
idx = np.arange(M) / M * 2
idx = [i if i <= 1 else i-2 for i in idx]
angle = np.arccos(idx) / np.pi
seq_angle = np.arange(M)/M

# 测试层次码本码字组合后的响应情况
DFT_codebook_temp = 1 / np.sqrt(M) * np.exp(
    2j * np.pi * np.outer(np.arange(M), (np.arange(M) + (1-M)/2)) / M)
DFT_codebook2 = np.zeros((M, int(M/2)), dtype=np.complex128)
for i in range(0, M, 2):
    idx = i // 2
    DFT_codebook2[:, idx] = DFT_codebook_temp[:, i] * np.exp(1j * np.pi * (-1 + 1/M) * (i+1)) + DFT_codebook_temp[:, i+1] * np.exp(1j * np.pi * (-1 + 1/M) * (i+2))
DFT_codebook2 = 1/np.sqrt(2) * DFT_codebook2.conj()

test = [DFT_codebook_temp[:, i] * np.exp(1j * np.pi * (-1 + 1/M) * (i+1)) for i in range(16, 19)]
DFT_combine_4 = np.sum(test, axis=0) * 1/np.sqrt(3)
DFT_combine_4 = DFT_combine_4.conj()

# print(DFT_combine_4.conj().T @ DFT_combine_4)

L = 200
a= np.zeros((L,M), dtype=complex)
theta = [np.pi/(L-1) * i for i in range(L)]
for i, t in enumerate(theta):
    a[i, :] = 1 / np.sqrt(M) * np.exp(-2j * np.pi * np.arange(M) / 2 * np.cos(t))

angel_response1 = np.abs(a.conj() @ DFT_codebook_temp[:, 16].conj())
angel_response12 = np.abs(a.conj() @ DFT_codebook_temp[:, 17].conj())
angel_response13 = np.abs(a.conj() @ DFT_codebook_temp[:, 18].conj())
# angel_response14 = np.abs(a.conj() @ DFT_codebook_temp[:, 19].conj())
# angel_response2 = np.abs(a.conj() @ DFT_codebook2[:, 8])
# angel_response21 = np.abs(a.conj() @ DFT_codebook2[:, 9])
# angel_response22 = np.abs(a.conj() @ DFT_codebook2[:, 10])
angel_response24 = np.abs(a.conj() @ DFT_combine_4)

plt.figure()
plt.plot([t/np.pi for t in theta], angel_response1, label="DFT")
plt.plot([t/np.pi for t in theta], angel_response12)
plt.plot([t/np.pi for t in theta], angel_response13)
# plt.plot([t/np.pi for t in theta], angel_response14)
plt.plot([t/np.pi for t in theta], angel_response24, label="combine")
# plt.plot([t/np.pi for t in theta], angel_response21)
# plt.plot([t/np.pi for t in theta], angel_response22)
plt.grid(True)
plt.title(f"codebook angel response")
plt.xlabel("angle/Π")
plt.legend()
plt.show()

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
# plt.title(f"average angel response(r={r})")
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