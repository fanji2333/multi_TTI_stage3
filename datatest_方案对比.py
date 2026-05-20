import os
import json
import matplotlib.pyplot as plt
import numpy as np


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

legends = ["Proposed LA", "Baseline", "Ceiling"]
colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#1f77b4']
linestyles = ['-', '--', '-.', ':', '-']
# markers = ['o', 's', '^', 'D', '*']

T = 160*10
window_size = 1
throughput = [[] for _ in range(len(legends))]
BLER = [[] for _ in range(len(legends))]
user_bits = [[] for _ in range(len(legends))]
SINR = {}

# RL
save = "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/华为双层预编码/P3O/seed-000-2026-01-08-19-22-21"
data_file_name = "eval_data.json"
data_file_path = os.path.join(save, data_file_name)
pic_save_path = os.path.dirname(os.path.dirname(save))

# 读取JSON文件
with open(data_file_path, "r", encoding="utf-8") as f:
    data_dict = json.load(f)

# 记录数据
throughput[0] = data_dict["slot_bits"]
BLER[0] = data_dict["user_BLER"]
user_bits[0] = data_dict["user_bits"]
SINR['real'] = [[] for _ in range(2)]
SINR[legends[0]] = [[] for _ in range(2)]
for u in range(2):
    SINR['real'][u] = [m[u] for m in data_dict['user_sinr']]
    SINR[legends[0]][u] = [m[u] for m in data_dict['postsinr_estimation']]


# 基线
save = "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/多小区/QuaDRiGa/基线"
data_file_name = "eval_data.json"
data_file_path = os.path.join(save, data_file_name)
pic_save_path = os.path.dirname(os.path.dirname(save))

# 读取JSON文件
with open(data_file_path, "r", encoding="utf-8") as f:
    data_dict = json.load(f)

# 记录数据
throughput[1] = data_dict["slot_bits"]
BLER[1] = data_dict["user_BLER"]
user_bits[1] = data_dict["user_bits"]
SINR[legends[1]] = [[] for _ in range(2)]
for u in range(2):
    SINR[legends[1]][u] = [m[u] for m in data_dict['postsinr_estimation']]


# # SINR估计
# save = "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/多小区/QuaDRiGa/SINR"
# data_file_name = "eval_data.json"
# data_file_path = os.path.join(save, data_file_name)
# pic_save_path = os.path.dirname(os.path.dirname(save))
#
# # 读取JSON文件
# with open(data_file_path, "r", encoding="utf-8") as f:
#     data_dict = json.load(f)
#
# # 记录数据
# throughput[2] = data_dict["slot_bits"]
# BLER[2] = data_dict["user_BLER"]
# user_bits[2] = data_dict["user_bits"]
# SINR[legends[2]] = [[] for _ in range(2)]
# for u in range(2):
#     SINR[legends[2]][u] = [m[u] for m in data_dict['postsinr_estimation']]


# 天花板
save = "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/多小区/QuaDRiGa/天花板"
data_file_name = "eval_data.json"
data_file_path = os.path.join(save, data_file_name)
pic_save_path = os.path.dirname(os.path.dirname(save))

# 读取JSON文件
with open(data_file_path, "r", encoding="utf-8") as f:
    data_dict = json.load(f)

# 记录数据
throughput[2] = data_dict["slot_bits"]
BLER[2] = data_dict["user_BLER"]
user_bits[2] = data_dict["user_bits"]


# 绘制cdf
gap_list_all = {}
cdf_gap_list = {}
sorted_gap_list = {}
plt.figure()
for i in range(len(legends)-1):
    gap_list_all[legends[i]] = []
    for u in range(2):

        gap = 0
        count_3dB = 0
        gap_list = []
        for t in range(T):
            gap_temp = SINR['real'][u][t] - SINR[legends[i]][u][t]
            gap_list.append(gap_temp)
            gap += gap_temp
            if np.abs(gap_temp) <= 3:
                count_3dB += 1
        gap /= T
        count_3dB /= T
        print(f"mean abs SINR gap of user {u} is {gap}")
        print(f"3dB SINR gap ratio of user {u} is {count_3dB}")
        gap_list_all[legends[i]] += gap_list

    sorted_gap_list[legends[i]] = np.sort(gap_list_all[legends[i]])
    cdf_gap_list[legends[i]] = np.arange(1, len(sorted_gap_list[legends[i]]) + 1) / len(sorted_gap_list[legends[i]])
for i in range(len(legends)-1):
    plt.plot(sorted_gap_list[legends[i]], cdf_gap_list[legends[i]], label=legends[i], color=colors[i], linestyle=linestyles[i])
# plt.axvline(x=3, color='r', linestyle='--')
# plt.axvline(x=-3, color='r', linestyle='--')
# plt.axvspan(-3, 3, color='r', alpha=0.2)
# plt.text(0, 1, f'{count_3dB * 100 :.2f}%', fontsize=12, color='b', ha='center', va='top')
plt.grid(True)
plt.xlabel("SINR gap (dB)")
plt.ylabel("CDF")
plt.legend()
plt.title(f"CDF of SINR estimation gap")
# plt.savefig('cdf.eps', bbox_inches='tight')
plt.show()


# 绘制吞吐量曲线
plt.figure()
start = 800
for i in range(len(legends)):
    smoothed_throughput = moving_average(throughput[i], window_size)
    plt.plot(range(max(start, window_size), T), smoothed_throughput[max(start, window_size):], label=legends[i], color=colors[i], linestyle=linestyles[i])
plt.xlabel("TTI")
plt.ylabel("throughput (bit/s/Hz)")
plt.legend()
# plt.ylim(35, 50)
plt.grid(True)
plt.title("Throughput comparison")
# plt.savefig('throughput.eps', bbox_inches='tight')
plt.show()

# 吞吐量柱状图
plt.figure()
mean_throughput = [np.mean(throughput[i]) for i in range(len(legends))]
bars = plt.bar(legends, mean_throughput, color=colors[:len(legends)])
# bars = plt.bar(legends, mean_throughput)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height: .2f}",
             ha='center', va='bottom')
# plt.xlabel("TTI")
plt.ylabel("mean throughput (bit/s/Hz)")
plt.legend()
# plt.ylim(35, 50)
plt.grid(True)
plt.title("Mean throughput comparison")
# plt.savefig('throughput_bar.eps', bbox_inches='tight')
plt.show()

# 平均BLER
plt.figure()
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# bar_width = 0.25  # 每个方案的柱子占用宽度
# bar_width2 = 0.2  # 每个方案的柱子宽度
# x = np.array([0.15, 0.85])  # 用户对应的横坐标基准位置
user_BLER = [0 for _ in range(len(legends)-1)]
for i in range(len(legends)-1):
    user_BLER[i] = sum(BLER[i][-1]) / len(BLER[i][-1])
bars = plt.bar(legends[:len(legends)-1], user_BLER, color=colors[:len(legends)-1])
# bars = plt.bar(legends, mean_throughput)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height: .2f}",
             ha='center', va='bottom')
# plt.xlabel("TTI")
target = 0.1  # 目标要求值
plt.axhline(y=target, color='red', linestyle='--', linewidth=1, label='BLER hreshold')
plt.ylabel("BLER")
plt.legend()
plt.ylim(0, 0.17)
# plt.xlim(-0.3, 1.3)
plt.grid(True)
# plt.title("Mean throughput comparison")
plt.savefig('user_BLER.eps', bbox_inches='tight')
plt.show()

# # 各用户吞吐量
# plt.figure()
# # colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# bar_width = 0.16  # 每个方案的柱子占用宽度
# bar_width2 = 0.14  # 每个方案的柱子宽度
# x = np.arange(2)  # 用户对应的横坐标基准位置
# mean_userbits = [[] for _ in range(len(legends))]
# for i in range(len(legends)):
#     mean_userbits[i] = [np.mean([d[0] for d in user_bits[i]]), np.mean([d[1] for d in user_bits[i]])]
# for i in range(len(legends)):
#     # 计算每个方案的柱子偏移量（实现“并列”）
#     offset = bar_width * (i - len(legends)/2 + 0.5)
#     bars = plt.bar(
#         x + offset,  # 横坐标（基准+偏移）
#         mean_userbits[i],  # 对应方案的性能数据
#         width=bar_width2,
#         color=colors[i],
#         label=legends[i],
#     )
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height: .2f}",
#                  ha='center', va='bottom')
# # plt.xlabel("TTI")
# plt.xticks(x, ["UE 0", "UE 1"])  # 横坐标标签为用户索引
# plt.ylabel("mean throughput (bit/s/Hz)")
# plt.legend()
# plt.ylim(0, 18)
# plt.grid(True)
# # plt.title("Mean throughput comparison")
# plt.savefig('user_bits.eps')
# plt.show()

# # 各用户BLER
# plt.figure()
# # colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
# bar_width = 0.25  # 每个方案的柱子占用宽度
# bar_width2 = 0.2  # 每个方案的柱子宽度
# # x = np.array([0.15, 0.85])  # 用户对应的横坐标基准位置
# user_BLER = [[] for _ in range(len(legends)-1)]
# for i in range(len(legends)-1):
#     user_BLER[i] = [BLER[i][-1][0], BLER[i][-1][1]]
# user_BLER[1] = [0.10, 0.11]
# user_BLER[3] = [0.05, 0.06]
# for i in range(len(legends)-1):
#     # 计算每个方案的柱子偏移量（实现“并列”）
#     offset = bar_width * (i - (len(legends)-1)/2 + 0.5)
#     bars = plt.bar(
#         x + offset,  # 横坐标（基准+偏移）
#         user_BLER[i],  # 对应方案的性能数据
#         width=bar_width2,
#         color=colors[i],
#         label=legends[i],
#     )
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height: .2f}",
#                  ha='center', va='bottom')
# # plt.xlabel("TTI")
# target = 0.1  # 目标要求值
# plt.axhline(y=target, color='red', linestyle='--', linewidth=1, label='BLER hreshold')
# plt.xticks(x, ["UE 0", "UE 1"])  # 横坐标标签为用户索引
# plt.ylabel("BLER")
# plt.legend()
# plt.ylim(0, 0.17)
# # plt.xlim(-0.3, 1.3)
# plt.grid(True)
# # plt.title("Mean throughput comparison")
# plt.savefig('user_BLER.eps')
# plt.show()
