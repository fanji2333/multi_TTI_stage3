import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 比较新版SINR估计相比原版的性能
def moving_average(data, window_size):
    data_np = np.array(data)
    result = np.zeros_like(data_np, dtype = np.float64)
    for i in range(len(data_np)):
        if i < window_size:
            result[i]= np.mean(data_np[:i + 1])
        else:
            result[i]= np.mean(data_np[i - window_size+1:i + 1])
    return result

MCS_table = {
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

# 方案数据
method = "Multi-Cell Ceiling"
save = "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/多小区/QuaDRiGa/角度约束/天花板"
data_file_name = "eval_data.json"
data_file_path = os.path.join(save, data_file_name)
pic_save_path = os.path.dirname(os.path.dirname(save))

T = 160*10
window_size = 400

# 读取JSON文件
with open(data_file_path, "r", encoding="utf-8") as f:
    data_dict = json.load(f)

# 记录数据
throughput = data_dict["slot_bits"]
BLER = data_dict["user_BLER"]
mcs = [[] for _ in range(2)]
mcs_ideal = [[] for _ in range(2)]
mcs_to_SINR = [[] for _ in range(2)]
mcs_ideal_to_SINR = [[] for _ in range(2)]
SINR = [[] for _ in range(2)]
SINR_estimate = [[] for _ in range(2)]
SINR_estimate_raw = [[] for _ in range(2)]
user_layer = [[] for _ in range(2)]
gain = [[] for _ in range(2)]
interference = [[] for _ in range(2)]
interference_ICI = [[] for _ in range(2)]
for u in range(2):
    mcs[u] = [m[u] for m in data_dict['user_mcs']]
    mcs_to_SINR[u] = [MCS_table[str(m)][1] for m in mcs[u]]
    mcs_ideal[u] = [m[u] for m in data_dict['user_mcs_ideal']]
    mcs_ideal_to_SINR[u] = [MCS_table[str(m)][1] for m in mcs_ideal[u]]
    SINR[u] = [m[u] for m in data_dict['user_sinr']]
    SINR_estimate[u] = [m[u] for m in data_dict['postsinr_estimation']]
    SINR_estimate_raw[u] = [m[u] for m in data_dict['postsinr_estimation_raw']]
    user_layer[u] = [m[u] for m in data_dict["user_layer"]]
    gain[u] = [m[u] for m in data_dict["user_gain"]]
    interference[u] = [m[u] for m in data_dict["user_interference"]]
    interference_ICI[u] = [m[u] for m in data_dict["user_interference_ICI"]]

plt.figure()
smoothed_throughput = moving_average(throughput, window_size)
plt.plot(range(window_size, T), smoothed_throughput[window_size:])
plt.xlabel("TTI")
plt.ylabel("Throughput (bits/TTI)")
plt.legend()
plt.grid(True)
plt.title(f"{method} Throughput")
plt.show()


for u in range(2):
    plt.figure()
    plt.plot(range(T), mcs_to_SINR[u], label="real MCS")
    plt.plot(range(T), mcs_ideal_to_SINR[u], "--", label="ideal MCS")
    plt.xlabel("TTI")
    plt.ylabel("MCS to SINR (dB)")
    plt.legend()
    plt.grid(True)
    plt.title(f"{method} MCS to SINR comparison for user {u}")
    plt.show()

    plt.figure()
    plt.plot(range(T), SINR[u], label="real SINR")
    plt.plot(range(T), SINR_estimate[u], "--",
             label="estimated SINR")
    plt.plot(range(T), SINR_estimate_raw[u],
             label="estimated SINR without OLLA")
    plt.xlabel("TTI")
    plt.ylabel("SINR (dB)")
    plt.legend()
    plt.grid(True)
    plt.title(f"{method} SINR comparison for user {u}")
    plt.show()

    gap = 0
    for t in range(T):
        gap += np.abs(SINR[u][t] - SINR_estimate[u][t])
    gap /= T
    print(f"{method} mean SINR gap of user {u} is {gap}")

    plt.figure()
    plt.plot(range(T), user_layer[u])
    plt.xlabel("TTI")
    plt.ylabel("layer")
    plt.title(f"{method} layer set for user {u}")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(T), [m[0] for m in gain[u]], label="layer 1")
    valid_indices = []
    filtered_data2 = []
    for i, m in enumerate(gain[u]):
        if len(m) == 2:
            valid_indices.append(i)
            filtered_data2.append(m[1])
    plt.plot(valid_indices, filtered_data2, label="layer 2")
    plt.legend()
    plt.xlabel("TTI")
    plt.ylabel("dB")
    plt.title(f"{method} gain for user {u}")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(T), [m[0] for m in interference[u]], label="layer 1")
    valid_indices = []
    filtered_data2 = []
    for i, m in enumerate(interference[u]):
        if len(m) == 2:
            valid_indices.append(i)
            filtered_data2.append(m[1])
    plt.plot(valid_indices, filtered_data2, label="layer 2")
    plt.legend()
    plt.xlabel("TTI")
    plt.ylabel("dB")
    plt.title(f"{method} interference for user {u}")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(T), [m[0] for m in interference_ICI[u]], label="layer 1")
    valid_indices = []
    filtered_data2 = []
    for i, m in enumerate(interference_ICI[u]):
        if len(m) == 2:
            valid_indices.append(i)
            filtered_data2.append(m[1])
    plt.plot(valid_indices, filtered_data2, label="layer 2")
    plt.legend()
    plt.xlabel("TTI")
    plt.ylabel("dB")
    plt.title(f"{method} interference ICI for user {u}")
    plt.grid(True)
    plt.show()