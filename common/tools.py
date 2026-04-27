import numpy as np
import scipy.spatial as sci
from scipy.stats import gamma
import matplotlib.pyplot as plt
import os

import torch
from omnisafe.utils.tools import load_yaml
from omnisafe.utils.config import Config


class BiasCorrectedEWMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.beta = 1 - alpha  # 衰减因子
        self.raw_avg = 0  # 原始EWMA值
        self.corrected_avg = 0  # 校正后的值
        self.t = 0  # 时间步数

    def update(self, value):
        self.t += 1

        if self.t == 1:
            self.raw_avg = self.alpha * value
        else:
            self.raw_avg = self.beta * self.raw_avg + self.alpha * value

        # 偏差校正
        if self.beta < 1:
            correction = 1 / (1 - self.beta ** self.t)
            self.corrected_avg = self.raw_avg * correction
        else:
            self.corrected_avg = self.raw_avg

        return self.corrected_avg

    def reset(self):
        self.raw_avg = 0  # 原始EWMA值
        self.corrected_avg = 0  # 校正后的值
        self.t = 0  # 时间步数

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成位置
def generate_location(num_points, min_distance, x_range, y_range):
    points = []
    iter_times = 0

    while len(points) < num_points:
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        iter_times += 1

        if not points:
            points.append((x, y))
        else:
            dists = sci.distance.cdist([(x, y)], points)
            if np.all(dists >= min_distance):
                points.append((x, y))

        if iter_times > 500:
            print('points generation iter times is too large, consider change model cfg')
            print('try re-generation')
            points = []
            iter_times = 0

    return points


# 计算两点间距离
def get_dis(point1, point2, h1=None, h2=None):

    if h1 and h2:
        distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (h2 - h1) ** 2) ** 0.5
    else:
        distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

    return distance


def determine_rows_cols(N, W, H):
    best_ratio = float('inf')
    best_config = (0, 0)

    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols  # 计算所需的行数以至少包含N个点
        if rows * cols >= N:  # 确保有足够的格子容纳所有点
            current_ratio = abs(rows / cols - H / W)  # 计算当前比例与目标比例的偏差
            if current_ratio < best_ratio:  # 找到更接近目标宽高比的配置
                best_ratio = current_ratio
                best_config = (rows, cols)

    return best_config


# 将简单的对用户在子带上的调度转化为实际的三维调度矩阵及功率分配矩阵
def simple2action(schedule, B, U, K, dP, APs):

    # B = cfg["B"]
    # U = cfg["U"]
    # K = cfg["K"]
    # dP = cfg["dP"]
    # APs = cfg["APs"]

    assert schedule.shape[0] == K
    assert schedule.shape[1] == U

    # 简化AP调度，只考虑用户在子带上的调度，一旦用户调度到某一子带，即让所有AP共同服务这一用户
    zeta = np.zeros((K, U, B))
    for k in range(K):
        for u in range(U):
            for b in range(B):
                zeta[k, u, b] = schedule[k, u]

    # 简化功率分配，对于同一AP而言，将其总功率均摊到所有需要服务的用户上
    P = np.zeros((K, U, B))
    for b in range(B):
        user_num = np.sum(zeta[:, :, b])
        p = int(APs[b]["max P"] // user_num)
        P[:, :, b] = p * zeta[:, :, b]

    P = P * dP

    return zeta, P


# 计算gamma分布的上α分位数
def calculate_d0(alpha, lambda_, N):

    d0 = gamma.ppf(1 - alpha, a=N, scale=1/lambda_)
    return d0

# 将tensor转换为二进制调度矩阵
def tensor2binary(action: torch.Tensor, simple_env: bool, B, U, K):
    if simple_env:
        # tensor维度应为K*U
        assert action.shape[0] == K * U
        # 重整形状
        new_action = action.reshape(K, U)
        if action.is_cuda:
            new_action = new_action.cpu().numpy()
        else:
            new_action = new_action.numpy()
        # 二值化
        new_action[new_action > 0] = 1
        new_action[new_action <= 0] = 0
        # 扩展得到三维矩阵
        expanded_matrix = np.expand_dims(new_action, axis=-1)
        new_action = np.repeat(expanded_matrix, B, axis=-1)
    else:
        # tensor维度应为K*U*B
        assert action.shape[0] == K * U * B
        # 重整形状
        new_action = action.reshape(K, U, B)
        if action.is_cuda:
            new_action = new_action.cpu().numpy()
        else:
            new_action = new_action.numpy()
        # 二值化
        new_action[new_action > 0] = 1
        new_action[new_action <= 0] = 0

    return new_action

def tensorReshape(action: torch.Tensor, simple_env: bool, B, U, K):
    if simple_env:
        # tensor维度应为K*U
        assert action.shape[0] == K * U
        # 重整形状
        new_action = action.reshape(K, U)
        if action.is_cuda:
            new_action = new_action.cpu().numpy()
        else:
            new_action = new_action.numpy()
        # 扩展得到三维矩阵
        expanded_matrix = np.expand_dims(new_action, axis=-1)
        new_action = np.repeat(expanded_matrix, B, axis=-1)
    else:
        # tensor维度应为K*U*B
        assert action.shape[0] == K * U * B
        # 重整形状
        new_action = action.reshape(K, U, B)
        if action.is_cuda:
            new_action = new_action.cpu().numpy()
        else:
            new_action = new_action.numpy()

    return new_action

# 针对调度情况，按照平均分配原则完成功率分配
def schedule2P(schedule: np.ndarray, dP, max_P):
    # 简化功率分配，对于同一AP而言，将其总功率均摊到所有需要服务的用户上
    P = schedule.copy()
    for b in range(P.shape[2]):

        candidate = np.argwhere(schedule[:, :, b])
        candidate_num = len(candidate)
        # 若待分配数大于可分配总数
        if candidate_num > max_P:
            P[:, :, b] = 0 * schedule[:, :, b]
            selected = candidate[np.random.choice(candidate_num, max_P, replace=False)]
            for position in selected:
                k, u = position
                P[k, u, b] += 1
        # 若可分配总数能够满足分配需求
        else:
            p = int(max_P // candidate_num) if candidate_num != 0 else 0
            P[:, :, b] = p * schedule[:, :, b]
            # rest = int(max_P % candidate_num) if candidate_num != 0 else 0
            # selected = candidate[np.random.choice(candidate_num, rest, replace=False)]
            # for position in selected:
            #     k, u = position
            #     P[k, u, b] += 1

    P = P * dP

    return P


def plot(data: list, axisx: range, title: str, directory: str = None):

    plt.figure()
    need_label = False

    if isinstance(data[0], list):
        need_label = True
        for idx in range(len(data[0])):
            data_sub = [subdata[idx] for subdata in data]
            data_smoothed = smooth(data_sub)
            plt.plot(axisx, data_sub, label=f"User{idx}")

    else:
        # data_smoothed = smooth(data)
        plt.plot(axisx, data)
        # plt.plot(axisx, data_smoothed)

    plt.title(title)
    if title == 'epoch ave return' or title == 'epoch ave cost':
        plt.xlabel('epochs')
    else:
        plt.xlabel('slots')

    if title in ['tot bits per slot', 'Users queue condition']:
        plt.ylabel('bits')
    elif title == 'tot Bw used per slot':
        plt.ylabel('Hz')
    elif title == 'SE per slot':
        plt.ylabel('bits/s/Hz')

    if need_label:
        plt.legend()
    plt.grid(True)

    if directory:
        # 确保目录存在
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 生成唯一的文件名，基于当前时间
        filename = f"{title}.png"
        filepath = os.path.join(directory, filename)

        # 保存图像到指定路径
        plt.savefig(filepath)

    plt.show()


def plot_hist(data, title: str, xlabel: str = "SINR/dB"):

    category = len(data)

    for i in range(category):
        plt.hist(data[i], bins=np.arange(-165, -125, 0.25), density=True, alpha=0.7, label=f'interference num {i}')

    plt.title(title)
    plt.xlabel(xlabel)
    # plt.ylabel("Frequency")
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.legend()

    plt.show()


def plot_bar(data, x_data: list, title: str, xlabel: str = "MCS order", directory: str = None):

    if x_data is None:
        x_data = [i+1 for i in range(len(data))]
    plt.bar(x_data, data, alpha=0.7)

    plt.title(title)
    plt.xlabel(xlabel)
    # plt.ylabel("Frequency")
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.legend()

    if directory:
        # 确保目录存在
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 生成唯一的文件名，基于当前时间
        filename = f"{title}.png"
        filepath = os.path.join(directory, filename)

        # 保存图像到指定路径
        plt.savefig(filepath)

    plt.show()


def smooth(data):
    window_size = 20  # Define the window size for the moving average

    # Compute the sliding window average
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Align the smoothed data length with the original data
    smoothed_data = np.concatenate(([np.nan] * (window_size - 1), smoothed_data))

    return smoothed_data

def get_default_kwargs_yaml(algo: str) -> Config:
    """Get the default kwargs fro ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name. Make
        sure your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        algo (str): The algorithm name.
        env_id (str): The environment name.
        algo_type (str): The algorithm type.

    Returns:
        The default kwargs.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, '..', 'configs', f'{algo}.yaml')
    print(f'Loading {algo}.yaml from {cfg_path}')
    kwargs = load_yaml(cfg_path)
    default_kwargs = kwargs['defaults']
    # env_kwargs = kwargs[env_id] if env_id in kwargs else None

    default_kwargs = Config.dict2config(default_kwargs)

    # if env_kwargs is not None:
    #     default_kwargs.recurisve_update(env_kwargs)

    return default_kwargs


# def plot_bar(data: list, title: str):
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
#
#     N = len(data)
#     bar_width = 0.2
#     bar_start = 0.3
#     labels = ['拉格朗日安全强化学习', '固定乘子0.1', '固定乘子0.01']
#
#     if type(data[0]) != list :
#         for i in range(N):
#             plt.bar(i * bar_start, data[i], width=bar_width, label=labels[i])
#         plt.xticks([])
#     else:
#         U = len(data[0])
#         x = np.arange(U)
#         for i in range(N):
#             plt.bar(x + i * bar_start, data[i], width=bar_width, label=labels[i])
#         plt.xticks(x + bar_start * (N - 1) / 2, [f'用户 {i + 1}' for i in range(U)])
#
#     plt.grid(axis='y')
#     plt.title(title)
#     plt.legend()
#
#     if title == '各用户频谱效率':
#         plt.ylim(0, 20)
#         plt.ylabel('bits/s/Hz')
#     if title == '三种算法总频谱效率对比':
#         plt.ylabel('bits/s/Hz')
#     if title == '各用户时延约束违反率':
#         plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.01))
#         plt.grid(axis='y', which='both', color='gray', linestyle='-', alpha=0.5)  # 设置主要网格线
#         plt.grid(axis='y', which='minor', color='lightgray', linestyle=':', alpha=0.5)  # 设置次要网格线
#
#     plt.show()
