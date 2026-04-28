from common.evaluator import Evaluator
from common.tools import get_default_kwargs_yaml
import matplotlib.pyplot as plt
import torch


# contents = torch.load('runs/低复杂度确定性/P3O/seed-000-2025-05-10-11-17-47_P3O第一次完整训练，训练时效果好，但测试时无法复现，似乎是obs归一化参数未保存导致的/torch_save/epoch-1200.pt', map_location='cpu', weights_only=True)
# print(contents.keys())

T = 160*10

# 验证所提算法训练模型
save = \
        "/home/fj24/26_4_Huawei_multiTTI_stage3/runs/华为双层预编码/P3O/seed-000-2026-01-08-19-22-21/torch_save/epoch-1400.pt"
cfgs = get_default_kwargs_yaml('P3O')
eval_obj = Evaluator(cfgs, save)
data_dict = eval_obj.evaluate(T, need_plot=True)

# eval_obj.plot_pos()

user_num = len(data_dict['user_sinr'][0])
user_sinr = [[] for _ in range(user_num)]
user_mcs = [[] for _ in range(user_num)]
user_mcs_ideal = [[] for _ in range(user_num)]
postsinr_estimation = [[] for _ in range(user_num)]
postsinr_estimation_raw = [[] for _ in range(user_num)]

T2 = 160
idx = 5
for u in range(user_num):
    user_sinr[u] = [sinr[u] for sinr in data_dict['user_sinr']]
    user_mcs[u] = [mcs[u] for mcs in data_dict['user_mcs']]
    user_mcs_ideal[u] = [mcs[u] for mcs in data_dict['user_mcs_ideal']]
    postsinr_estimation[u] = [postsinr[u] for postsinr in data_dict['postsinr_estimation']]
    postsinr_estimation_raw[u] = [postsinr_raw[u] for postsinr_raw in data_dict['postsinr_estimation_raw']]

    plt.figure()
    plt.plot(range(T2*idx, T2*(idx+1)), user_sinr[u][T2*idx:T2*(idx+1)], label="real sinr")
    plt.plot(range(T2*idx, T2*(idx+1)), postsinr_estimation[u][T2*idx:T2*(idx+1)], label="estimated sinr")
    plt.plot(range(T2*idx, T2*(idx+1)), postsinr_estimation_raw[u][T2*idx:T2*(idx+1)], label="estimated sinr (without OLLA)")
    # plt.plot(user_mcs[u], label="mcs")
    plt.grid(True)
    plt.legend()
    plt.title(f"sinr condition of user {u}")
    plt.show()

for u in range(user_num):
    plt.figure()
    plt.plot(range(T2*idx, T2*(idx+1)), user_mcs[u][T2*idx:T2*(idx+1)], label="L2 MCS")
    plt.plot(range(T2*idx, T2*(idx+1)), user_mcs_ideal[u][T2*idx:T2*(idx+1)], label="ideal MCS")
    plt.grid(True)
    plt.legend()
    plt.title(f"MCS condition of user {u}")
    plt.show()
