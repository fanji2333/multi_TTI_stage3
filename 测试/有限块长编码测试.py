import numpy as np
from scipy import stats
import scipy.special as sp
import matplotlib.pyplot as plt

def get_rate(sinr, bler):
    n = 12*14   # 码长不变
    sinr = 10**(sinr/10)
    return np.log2(1 + sinr) - stats.norm.ppf(1-bler) * np.log2(np.e) * np.sqrt((1 - 1/((1+sinr)**2))/n)

def get_bler(sinr, R):
    n = 12*14
    C = np.log2(1 + sinr)
    sqrt_V = np.log2(np.e) * np.sqrt(1 - 1/((1+sinr)**2))
    x = sp.erf(((C - R) / (sqrt_V / np.sqrt(n))) / np.sqrt(2))
    return 0.5 * (1 - x)

# sinr_dB = [-6.0 + i*1.3 for i in range(28)]
# sinr_dB = [-6.50, -4.00, -2.60, -1.00, 1.00, 3.00, 6.60, 10.00, 11.40, 11.80, 13.00, 13.80, 15.60, 16.80, 17.60]
# sinr_dB = [-5.65, -3.55, -1.50, 0.50, 2.45, 4.40, 5.40, 6.30, 7.25, 8.30, 8.95, 10.15, 11.25, 12.10, 13.15, 14.05, 15.10, 16, 17]
# R = [0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32, 3.90, 4.52, 5.12, 5.55]
R = [0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.6953, 1.9141, 2.1602, 2.4063, 2.5703, 2.7305, 3.0293, 3.3223,
     3.6094, 3.9023, 4.2129, 4.5234, 4.8164, 5.1152, 5.3320, 5.5547, 5.8906, 6.2266, 6.5703, 6.9141, 7.1602, 7.4063]

k = 27
sinr_dB = [0.01 * i for i in range(1600,5000)]
min_dif = 100
best_sinr = 0
for i, sinr_temp in enumerate(sinr_dB):
    sinr = 10 ** (sinr_temp / 10)
    bler = get_bler(sinr, R[k])
    dif = bler - 0.1
    # if dif < min_dif:
    #     min_dif = dif
    #     best_sinr = sinr_temp
    if bler < 0.1:
        break
print(f"best SINR is {sinr_temp:.2f} dB with BLER = {bler:.2f}, diff is {dif}")

# for i, sinr_temp in enumerate(sinr_dB):
#     sinr = 10 ** (sinr_temp/10)
#     bler = []
#     for r in R:
#         bler.append(get_bler(sinr, r))
#     plt.figure()
#     plt.plot(R, bler, label=f"sinr = {sinr_temp}dB")
#     plt.grid()
#     plt.xlabel('R')
#     plt.ylabel('bler')
#     plt.title(f'bler in various coding rate, SINR = {sinr_temp:.3f}dB, match MCS {i}')
#     # plt.legend()
#     plt.show()

# bler_t = [-3, -2, -1, -0.75, -0.5]
# sinr = [i for i in range(-6, 7)]
# plt.figure()
# for bler in bler_t:
#     R = [get_rate(gamma, 10**bler)*(1-10**bler) for gamma in sinr]
#     plt.plot(sinr, R, label=f"BLER={10**bler}")
# plt.grid(True)
# plt.ylabel("rate(bits/symbol)")
# plt.xlabel("SINR(dB)")
# plt.legend()
# plt.title("channel coding rate with different SINR in various BLER")
# plt.show()
#
# # bler_t = [0.001, 0.01, 0.1, 0.15, 0.2, 0.25]
# bler_t = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.75, -0.5]
# sinr = [i for i in range(-6, 31, 6)]
# for gamma in sinr:
#     plt.figure()
#     plt.grid(True)
#     R = [get_rate(gamma, 10**bler)*(1-10**bler) for bler in bler_t]
#     plt.plot([10**bler for bler in bler_t], R, label=f"SINR={gamma}dB")
#     plt.ylabel("rate expectation(bits/symbol)")
#     plt.xlabel("BLER")
#     # plt.legend()
#     plt.title(f"rate expectation with different BLER (SINR={gamma}dB)")
#     plt.show()