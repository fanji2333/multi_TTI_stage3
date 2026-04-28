import copy
# from operator import truediv

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch
# from common.tools import get_dis, generate_location, tensor2binary, tensorReshape, schedule2P, calculate_d0, determine_rows_cols
from common.tools import get_dis, determine_rows_cols, sigmoid

from omnisafe.utils.config import Config
from omnisafe.common.normalizer import Normalizer

from typing import List
from collections import deque


class Packet:

    def __init__(self, packetsize, maxdelay):
        self._packetsize = packetsize
        self._maxdelay = maxdelay
        self._delay = 0
        self._time_out = False

    def update(self):
        self._delay += 1
        if self._delay > self._maxdelay:
            self._time_out = True
        return self._time_out

    def get_delay(self):
        return self._delay


class User:

    def __init__(self, user_id, packet_size, max_delay, arrival_rate, delay_req, err_rate, K, shadow = None, location = None):
        self.user_id = user_id
        self.location = location
        self.shadow = shadow
        self.packet_size = packet_size
        self.max_delay = max_delay
        self.arrival_rate = arrival_rate
        self.delay_req = 10 ** (-delay_req)
        self.log_delay_req = np.log(self.delay_req)
        self.err_rate = err_rate
        self.K = K
        self.packet_queue: List[Packet] = []
        self.backlog = 0
        self.delay_queue = []
        self.transmitted_packets = 0
        # self.schedule_rate = 0
        self.totpackets = 0
        self.totfail = 0
        self.windowed_mean_rate = 0     # 滑窗平均速率(bits / slot)
        self.tc = 1 / 10
        self.window = 10
        self.rate_window = deque(maxlen=self.window)
        self.theta_choose = [1, 1.2, 1.4, 1.6, 1.8, 2.2, 2.6, 3]
        self.user_metric = 0            # EXP算法的metric，不包括实时速率
        self.delay_metric = 0           # 最大时延占时延界的比例，用于计算EXP metric
        self.gain = [0] * self.K  # 实时增益
        self.rate_estimated = 0         # 理论计算所需的稳定服务速率(packets / slot)


    def packet_arrival(self):
        arrive_num = np.random.poisson(self.arrival_rate)
        for _ in range(arrive_num):
            self.packet_queue.append(Packet(self.packet_size, self.max_delay))
            self.delay_queue.append(0)
        self.backlog = len(self.packet_queue) * self.packet_size
        assert len(self.packet_queue) == len(self.delay_queue), \
            (f'delay queue must change with packet queue, now delay queue is {len(self.delay_queue)}, '
             f'packet queue is {len(self.packet_queue)}')
        return arrive_num

    def transmit_update_queue(self, packet_num):
        self.packet_queue = self.packet_queue[packet_num:]
        self.backlog = len(self.packet_queue) * self.packet_size
        self.delay_queue = self.delay_queue[packet_num:]
        assert len(self.packet_queue) == len(self.delay_queue), \
            (f'delay queue must change with packet queue, now delay queue is {len(self.delay_queue)}, '
             f'packet queue is {len(self.packet_queue)}')
        self.transmitted_packets += packet_num
        self.totpackets += packet_num

    def update_delay_queue(self):

        # 超时的包丢弃
        time_out_packet = []
        for packet in self.packet_queue:
            if packet.update():
                time_out_packet.append(packet)
                self.totpackets += 1
                self.totfail += 1
        new_queue = [pkt for pkt in self.packet_queue if pkt not in time_out_packet]
        self.packet_queue = new_queue
        # 更新时延队列
        self.delay_queue = [pkt.get_delay() for pkt in self.packet_queue]
        # 更新bit队列
        self.backlog = len(self.packet_queue) * self.packet_size
        return len(time_out_packet)

    def transmit(self, bits, slot):
        # 仿真数据包传输过程
        rest = self.backlog - bits
        if rest >= 0:
            # 承载能力不足以清空队列，将尽量多的包组块发送
            outs = int(bits // self.packet_size)
            tx_bits = outs * self.packet_size  # 过剩的承载能力不发送半截的包

            # 根据块错误率决定是否重传，认为ACK在下个时隙就可到达，因此成功的包在这里可以直接当成发送成功，
            # 失败的包在这里就当成不出队列处理，效果跟下个时隙接收ACK后处理等价
            block_fail = np.random.rand()
            if block_fail > self.err_rate:
                # 传输成功，从队列中除去包
                self.transmit_update_queue(outs)
            else:
                # 块错误，传输失败，队列不变
                outs = 0

            # self.schedule_rate = (self.schedule_rate * (slot - 1) + outs) / slot
        else:
            # 承载能力足以清空队列
            tx_bits = self.backlog
            outs = int(tx_bits // self.packet_size)

            block_fail = np.random.rand()
            if block_fail > self.err_rate:
                # 传输成功，从队列中除去包
                self.transmit_update_queue(outs)
            else:
                # 块错误，传输失败，队列不变
                outs = 0

            # schedule_packets_cap.append(outs)
            # self.schedule_rate = (self.schedule_rate * (slot - 1) + outs) / slot

        # # 更新时延
        # time_out_num = self.update_delay_queue()

        # return time_out_num, tx_bits
        return tx_bits

    def update_rate_window(self, tx_bits):
        self.rate_window.append(tx_bits)
        self.windowed_mean_rate = self.windowed_mean_rate * (1 - self.tc) + self.tc * tx_bits
        # length = sum(1 for r in self.rate_window if r != 0)
        # self.windowed_mean_rate = sum(self.rate_window) / max(1, length)
        # self.windowed_mean_rate = sum(self.rate_window) / max(1, len(self.rate_window))

    def calculate_rate_estimation(self):
        # 执行保障速率估计
        # 由于在每个slot都用数值方法搜索方程解复杂度太高，改为在0队列解基础上缩放，选取最接近实际解的缩放因子
        theta_star = np.log(1 - self.log_delay_req / (self.max_delay * self.arrival_rate))
        best_theta = theta_star
        best_result = 10
        # 搜索合适的theta取值
        for ratio in self.theta_choose:
            theta = theta_star * ratio
            result = self.theta_equition(theta)
            if abs(result) < abs(best_result):
                best_result = result
                best_theta = theta
        # 计算最合适theta取值下的保障速率
        self.rate_estimated = self.rate_estimation(best_theta)
        # 使用历史平均服务速率缩放速率要求，期望以此弥补用户调度时的时间分集所带来的稳定速率要求不满足情况
        # if self.windowed_mean_rate > 0:
        #     self.rate_estimated = (self.rate_estimated ** 2) / (self.windowed_mean_rate / self.packet_size)
        return self.rate_estimated

    def rate_estimation(self, theta):
        return self.arrival_rate * (np.exp(theta) - 1) / theta

    def theta_equition(self, theta):
        return (self.max_delay * self.arrival_rate * np.exp(theta) - theta * len(self.packet_queue)
                + self.log_delay_req - self.max_delay * self.arrival_rate)

    def update_delay_metric(self):
        if not self.delay_queue:
            self.delay_metric = 0
        else:
            self.delay_metric = max(self.delay_queue) / self.max_delay
        return self.delay_metric

    def clear_gain(self):
        self.gain = [0] * self.K

    def reset(self):
        self.packet_queue: List[Packet] = []
        self.backlog = 0
        self.delay_queue = []
        self.transmitted_packets = 0
        # self.schedule_rate = 0
        self.totpackets = 0
        self.totfail = 0
        self.windowed_mean_rate = 0
        self.rate_window.clear()
        self.user_metric = 0
        self.delay_metric = 0
        self.gain = [0] * self.K


class O_RU:

    def __init__(self, O_RU_id, max_P, location, user_num, PRB_num, Antenna_num, codebook):
        self.O_RU_id = O_RU_id
        self.max_P = max_P
        self.location = location
        self.user_num = user_num
        self.PRB_num = PRB_num
        self.Antenna_num = Antenna_num
        # self.codebook = codebook
        # self.codebook_use = np.zeros((self.PRB, self.Antenna))
        # self.codebook_has_scheduled = [False] * self.PRB
        self.user_connection = []
        self.user_precoding = np.zeros((self.PRB_num, self.user_num, self.Antenna_num), dtype=complex)

    def reset(self):
        self.user_connection = []
        self.user_precoding = np.zeros((self.PRB_num, self.user_num, self.Antenna_num), dtype=complex)

    # def reset_codebook(self):
    #     self.codebook_use = np.zeros((self.PRB, self.Antenna))
    #     self.codebook_has_scheduled = [False] * self.PRB


class EDU:
    def __init__(self, O_RUs: List[O_RU]):
        self.O_RUs = O_RUs
        self.D = {}                               # 每个用户的连接矩阵，key为用户id，大小为LN * LN
        self.user_connection = []
        self.L = len(self.O_RUs)                  # EDU连接的总O-RU数
        self.N = self.O_RUs[0].Antenna_num        # 每个O-RU上的天线数
        self.EDU_H = {}                           # EDU处收集到的信道信息，分每个用户存储，key为用户id，大小为PRB数 * EDU总天线数
        # self.O_RUs_ids = []
        #
        # for o_ru in self.O_RUs:
        #     self.O_RUs_ids.append(o_ru.O_RU_id)

    # 在完成用户与O-RU配对以后，将用户与EDU相应配对，并且整理出与本EDU中O-RU的连接矩阵D
    def collect_users(self):
        L = self.L
        N = self.N
        for i, o_ru in enumerate(self.O_RUs):
            for user in o_ru.user_connection:
                # 如果用户与O-RU连接但尚未收录进EDU，则创建其连接矩阵D
                if user not in self.user_connection:
                    self.user_connection.append(user)
                    matrix = np.zeros((L*N, L*N))
                    # 与EDU中的第i个O-RU连接
                    matrix[N*i:N*(i+1), N*i:N*(i+1)] = np.eye(N)
                    self.D[user.user_id] = matrix
                    self.EDU_H[user.user_id] = np.zeros((self.O_RUs[0].PRB_num, L * N), dtype=complex)
                # 如果用户与O-RU连接且已在EDU中收录，则只需修改其连接矩阵D
                else:
                    self.D[user.user_id][N*i:N*(i+1), N*i:N*(i+1)] = np.eye(N)

    # 收集本EDU连接的用户对本EDU内的O-RU的信道，便于后期预编码处理
    def collect_channels(self, H):
        L = self.L
        N = self.N
        for user in self.user_connection:
            for i, o_ru in enumerate(self.O_RUs):
                self.EDU_H[user.user_id][:, N*i:N*(i+1)] = H[:, user.user_id, o_ru.O_RU_id]

    # EDU生成预编码，使用MMSE算法
    # TODO: 功率分配以用户为颗粒度，而非像以前那样细分到每个AP
    def calculate_precoding_vector(self, user:User, k, P, sigma):
        # 只有与EDU连接的用户才需要计算预编码
        if user in self.user_connection:
            temp_matrix = np.zeros((self.L * self.N, self.L * self.N), dtype=complex)
            for u in self.user_connection:
                h_col = self.EDU_H[u.user_id][k].reshape(-1, 1)
                # TODO: 文献中似乎利用了上下行信道互易性，在这里我直接定义了下行信道，
                #  因此计算时信道的取共轭情况与公式中相反，如果出现问题则再调整，
                #  此外这里也忽略了信道估计误差，所有用户发射功率不作区分
                h_temp = h_col.conj() @ h_col.T
                # temp_matrix += P[k, u.user_id] * self.D[user.user_id] @ h_temp @ self.D[user.user_id]
                temp_matrix += P * self.D[user.user_id] @ h_temp @ self.D[user.user_id]
            temp_matrix += sigma * np.eye(self.L * self.N)
            temp_matrix_inv = np.linalg.inv(temp_matrix)
            # precoding_vector = (P[k, user.user_id] * temp_matrix_inv @ self.D[user.user_id]
            #                     @ self.EDU_H[user.user_id][k].reshape(-1, 1).conj())
            precoding_vector = (P * temp_matrix_inv @ self.D[user.user_id]
                                @ self.EDU_H[user.user_id][k].reshape(-1, 1).conj())
            # TODO: 论文中公式到此为止，但应用到SINR计算时似乎还需要左乘连接矩阵D来表示O-RU与用户的连接情况
            precoding_vector = self.D[user.user_id] @ precoding_vector
            # 将计算得到的precoding vector重新拆为单O-RU发送的形式分别存储，便于后续调用
            for i, o_ru in enumerate(self.O_RUs):
                o_ru.user_precoding[k, user.user_id] = precoding_vector[self.N * i : self.N * (i+1)].flatten()

    def reset(self):
        self.D = {}
        self.user_connection = []
        self.EDU_H = {}
        for o_ru in self.O_RUs:
            o_ru.reset()


class UCDU:
    def __init__(self, O_RUs: List[O_RU], UEs: List[User], P, sigma, C, N):
        self.O_RUs = O_RUs
        self.UEs = UEs
        self.P = P
        self.sigma = sigma
        self.C = C
        self.N = N
        self.max_schedule = 5
        self.user_grouping_matrix = np.zeros((len(self.UEs), len(self.UEs)))
        self.user_interference_matrix = np.zeros((len(self.UEs), len(self.UEs)))
        self.connection_threshold = 1.5e-5
        self.interference_threshold = 1e-9
        self.UE_O_RU_allocation = [[] for _ in range(len(self.UEs))]
        self.UE_scheduled_indicator = np.zeros(len(self.UEs))

        # TODO: EDU在此实例化，后续如果需要拓展的话需要修改为更具可扩展性的代码
        #  现在配置是9个O-RU，设置3个EDU，每个EDU连接3个O-RU，按编号顺序连接
        self.EDUs = []
        for i in range(3):
            o_ru_list = []
            for j in range(3):
                o_ru_list.append(self.O_RUs[i*3 + j])
            self.EDUs.append(EDU(o_ru_list))

    def packets_arrival(self):
        # 数据包随机到达
        Queue_list = []
        for u in self.UEs:
            Queue_list.append(u.packet_arrival())
        return Queue_list

    def O_RU_allocation(self, Hl):
        # 按照大尺度衰落情况设置O_RU与用户关联
        O_RU_connection = Hl[0, :, :, 0]
        O_RU_connection = np.where(O_RU_connection > self.connection_threshold, 1, 0)
        for ap, col in zip(self.O_RUs, O_RU_connection.T):
            # ap.user_connection = col
            indices = [i for i, x in enumerate(col) if x == 1]
            # 根据索引提取元素
            for i in indices:
                ap.user_connection.append(self.UEs[i])
                self.UE_O_RU_allocation[i].append(ap)

    def user_choose(self):

        self.UE_scheduled_indicator = np.zeros(len(self.UEs))
        mean_delay_metric = 0
        for u in self.UEs:
            # 速率估计
            u.calculate_rate_estimation()
            mean_delay_metric += u.delay_metric
        mean_delay_metric = mean_delay_metric / len(self.UEs)

        metric_list = []
        for u in self.UEs:
            # TODO: metric设计暂时采用exp形式，但是比例公平部分的速率在这里是保障所需速率，应该不能反映channel情况，具体应如何修改还需要后续研究
            # 此处记录到User类里的metric只包括前半部分，不包括实时速率部分，以便底层使用
            # TODO: 测试发现比例公平的历史平均速率在初始时会出现较大的波动，导致调度不稳定？
            #  但如果去掉历史平均，则metric受实时速率影响过大，实时速率大的用户优势明显
            #  希望: 不要时间分集，要频率分集，这样才能让队列更稳定
            R_mean = 1 / u.windowed_mean_rate if u.windowed_mean_rate > 0 else 1
            # R_mean = 1
            # u.user_metric = R_mean * np.exp(u.delay_metric / (1 + mean_delay_metric))
            u.user_metric = np.exp(u.delay_metric / (1 + mean_delay_metric))
            metric_list.append(u.user_metric * u.rate_estimated * u.packet_size)

        # 取最大的前self.max_schedule个metric
        paired = [(value, idx) for idx, value in enumerate(metric_list)]
        paired_sorted = sorted(paired, key=lambda item: -item[0])
        top_metric_pairs = paired_sorted[:self.max_schedule]
        # top_metrics = [item[0] for item in top_metric_pairs]
        top_metric_users = [item[1] for item in top_metric_pairs]
        for u in top_metric_users:
            self.UE_scheduled_indicator[u] = 1
        return top_metric_users

    def generate_user_grouping_matrix(self, Hl):
        Hl = Hl[0, :, :, 0]        # 大尺度只与用户与O_RU的位置关系有关
        for u in self.UEs:
            for v in self.UEs:
                # 若用户v对用户u的总干扰小于阈值，则允许复用
                sum_interference = 0
                for b in self.UE_O_RU_allocation[v.user_id]:
                    sum_interference += Hl[u.user_id, b.O_RU_id] ** 2
                if sum_interference < self.interference_threshold:
                    self.user_grouping_matrix[u.user_id, v.user_id] = sum_interference * self.P / (self.N * self.C)
                self.user_interference_matrix[u.user_id, v.user_id] = sum_interference * self.P / (self.N * self.C)

    def resource_allocation(self, Hl, Hs, top_metric_users):

        resource_allocation = np.zeros((Hl.shape[0], len(self.UEs)))    # k*u PRB分配
        zeta = np.zeros((Hl.shape[0], len(self.UEs), len(self.O_RUs)))    # k*u*b 资源分配，只是在上面的基础上补上O_RU域关联，方便外层处理
        scheduled_users = [self.UEs[i] for i in top_metric_users]
        rate_matrix = np.zeros((Hl.shape[0], len(self.UEs) + 1))        # 记录各个PRB上的速率情况，每行最后位置为和速率
        # TODO:如果直接选择每个PRB上的primary user，可能出现所有primary user均为同一人的情况（因为上一slot中未接受调度的用户必然时延更大）
        #  因此，有必要在选择primary user的阶段也保障一定的用户间公平性，这里采取类轮询的方式，保证每个用户都能至少得到分配
        scheduled_users_round = copy.copy(scheduled_users)

        for k in range(Hl.shape[0]):
            # 找每个PRB上metric最大的用户作为primary user
            # metric采用用户选择时的EXP metric结合瞬时信道gain
            user_metric_list = []
            for u in scheduled_users_round:
                user_gain = self.calculate_user_gain(u, Hl, Hs, k)
                user_rate_estimated = self.user_rate_estimation(user_gain, u)
                user_metric_list.append(user_rate_estimated * u.user_metric)
            # primary user在轮询队列中选出，而不再从所有用户中选出
            primary_user = max(zip(scheduled_users_round, user_metric_list), key=lambda x: x[1])[0]
            # 在选择过primary user后，将其从轮询队列中剔除，直到所有用户都被选择过以后再开启下一轮轮询
            scheduled_users_round.remove(primary_user)
            if not scheduled_users_round:
                scheduled_users_round = copy.copy(scheduled_users)

            # 为每个primary user找复用用户直到预估和速率不再提升
            # 复用用户为primary user的可复用用户集中被上层选择的用户
            sub_users_list = list(self.user_grouping_matrix[primary_user.user_id, :])
            indices = [i for i, x in enumerate(sub_users_list) if x > 0 and i in top_metric_users]
            temp_indices = copy.copy(indices)
            # sub_users_list = [self.UEs[i] for i in indices]
            serve_user_list = [primary_user]
            primary_gain = self.calculate_user_gain(primary_user, Hl, Hs, k)
            max_sum_rate = self.user_rate_estimation(primary_gain, primary_user)
            while True:
                if temp_indices:
                    # 找干扰最小的用户作为复用选择
                    min_index = np.argmin(self.user_grouping_matrix[primary_user.user_id, temp_indices])
                    min_index = temp_indices[min_index]
                    best_multiplex_user = self.UEs[min_index]
                    temp_indices.remove(min_index)
                    serve_user_list.append(best_multiplex_user)
                    sum_rate = 0
                    for u in serve_user_list:
                        interference_list = copy.copy(serve_user_list)
                        interference_list.remove(u)
                        gain = self.calculate_user_gain(u, Hl, Hs, k)
                        sum_rate += self.user_rate_estimation(gain, u, interference_list, k, Hl, Hs)
                    if sum_rate > max_sum_rate:
                        max_sum_rate = sum_rate
                    else:
                        # 至少2用户复用？还是不强迫复用？
                        # if len(serve_user_list) > 2:
                        #     serve_user_list.pop()
                        serve_user_list.pop()
                        break
                else:
                    break

            # 初步确定PRB k上的复用情况，记录数据
            sum_rate = 0
            for u in serve_user_list:
                interference_list = copy.copy(serve_user_list)
                interference_list.remove(u)
                gain = self.calculate_user_gain(u, Hl, Hs, k)
                user_rate = self.user_rate_estimation(gain, u, interference_list, k, Hl, Hs)
                rate_matrix[k, u.user_id] = user_rate
                sum_rate += user_rate
            rate_matrix[k, -1] = sum_rate

        # 将各个PRB按和速率降序排列，依次确认PRB分配
        sorted_indices = np.argsort(rate_matrix[:, -1])[::-1]
        rate_total = np.zeros(len(self.UEs))
        abortion_users = []
        rate_target = np.zeros(len(self.UEs))
        for u in range(len(self.UEs)):
            rate_target[u] = self.UEs[u].rate_estimated * self.UEs[u].packet_size
        for k in sorted_indices:
            for u in range(len(self.UEs)):
                if rate_matrix[k, u] > 0:
                    if u not in abortion_users:
                        resource_allocation[k, u] = 1
                        for b in self.UE_O_RU_allocation[u]:
                            zeta[k, u, b.O_RU_id] = 1
                        rate_total[u] += rate_matrix[k, u]
                        # 若用户和速率已达标，则后续不再分配
                        if (u in top_metric_users and
                                rate_total[u] > self.UEs[u].rate_estimated * self.UEs[u].packet_size):
                            abortion_users.append(u)

                        # TODO: 在这里计算预编码，只有需要调度的用户才需要计算预编码，省一次遍历搜索
                        #  按论文中公式，MMSE预编码似乎在计算时就自带了功率配置，不需要另外显式指定
                        for edu in self.EDUs:
                            edu.calculate_precoding_vector(self.UEs[u], k, self.P/(self.N * self.C), self.sigma)
                    else:
                        continue

        gap = rate_total - rate_target

        return resource_allocation, zeta

    def calculate_user_gain(self, u, Hl, Hs, k):
        if u.gain[k] > 0:
            return u.gain[k]
        else:
            user_gain = 0
            for b in self.UE_O_RU_allocation[u.user_id]:
                user_gain += Hl[k, u.user_id, b.O_RU_id, 0] * np.linalg.norm(Hs[k, u.user_id, b.O_RU_id, :])
            user_gain = (user_gain ** 2) * self.P / (self.N * self.C)
            u.gain[k] = user_gain
            return user_gain

    # TODO: 重要！！！！
    # 用户的gain是实时的，存储进User实例只是为了避免重复计算，因此每个slot完成后应当清零
    def clear_user_gain(self):
        for u in self.UEs:
            u.clear_gain()

    def user_rate_estimation(self, gain, u, interference_users=None, k=0, Hl=None, Hs=None):
        interference = 0
        # interference_0 = 0
        if interference_users is None:
            interference_users = []
        for v in interference_users:
            # interference += self.user_grouping_matrix[u.user_id, v.user_id]
            # interference += self.user_interference_matrix[u.user_id, v.user_id]
            temp = 0
            for o_ru in self.UE_O_RU_allocation[v.user_id]:
                temp += (Hl[k, u.user_id, o_ru.O_RU_id, 0] *
                         (np.conj(Hs[k, v.user_id, o_ru.O_RU_id]) @ Hs[k, u.user_id, o_ru.O_RU_id]) /
                         np.linalg.norm(Hs[k, v.user_id, o_ru.O_RU_id]))
            interference += (np.abs(temp) ** 2) * self.P / (self.N * self.C)

        # print(f"old interference: {interference_0}, new interference: {interference}")
        return np.log2(1 + gain / (interference + self.sigma)) * self.C * self.N

    def rate_fix(self, rate):
        for i, u in enumerate(self.UEs):
            u.rate_estimated += rate[i]
            # u.rate_estimated *= 1.6

    def get_rate_estimated(self):
        rate_list = []
        for u in self.UEs:
            rate_list.append(u.rate_estimated)
        return rate_list

    def reset(self, Hl):
        for user in self.UEs:
            user.reset()

        self.user_grouping_matrix = np.zeros((len(self.UEs), len(self.UEs)))
        self.user_interference_matrix = np.zeros((len(self.UEs), len(self.UEs)))
        self.UE_O_RU_allocation = [[] for _ in range(len(self.UEs))]
        self.UE_scheduled_indicator = np.zeros(len(self.UEs))

        # O-RU的reset由EDU来完成
        for edu in self.EDUs:
            edu.reset()

        self.O_RU_allocation(Hl)

        for edu in self.EDUs:
            edu.collect_users()

        self.generate_user_grouping_matrix(Hl)

        self.clear_user_gain()


class Environment:

    def __init__(self, cfg: Config, device):

        self._cfg = cfg
        self._device = device

        # 环境基本设置
        self._B = cfg.B              # O_RU数量
        self._U = cfg.U              # 用户数量
        self._Antenna = cfg.Antenna  # O_RU天线数量
        self._Bw = cfg.Bw            # 单个子载波带宽(kHz)
        self._Bw_tot = cfg.Bw_tot    # 总带宽(MHz)
        self._K = cfg.K              # PRB数
        self._C = cfg.C              # 每个PRB上的子载波数
        self._N = cfg.N              # 每个时隙的OFDM符号数

        # 信道模型参数
        # self._d0 = cfg.d0            # 大尺度衰落参考距离
        # self._alpha = cfg.alpha      # 大尺度衰落路损因子
        # self._sigma = cfg.sigma_1 * (10 ** -cfg.sigma_2)     # 噪声功率
        self._fc = cfg.fc               # 中心频率(MHz)
        self._d0 = cfg.d0               # 大尺度衰落模型参数
        self._d1 = cfg.d1
        self._d2 = cfg.d2
        self._sigma_sh = cfg.sigma_sh   # shadow fading(dB)
        self._delta_sh = cfg.delta_sh   # shadow fading参数
        self._noise = 10 ** (cfg.noise/10 - 3)              # 噪声功率(W/Hz)

        self._F = self._K * self._C                         # 总载波数
        self._OFDM_t = 1 / (self._Bw * 1e3)                 # OFDM符号时长
        self._slot_t = cfg.t_slot                           # 时隙时长(ms)
        # self._f_carrier = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 for n in range(self._F)]   # 各载波频率(MHz)
        # self._f_subband = [self._fc + self._Bw_tot / 2 - (n + 1) * self._Bw * 1e-3 * self._C for n in range(self._K)]   # 各子带中心频率(MHz)
        self._H = None                                      # 信道矩阵
        self._Hl = None                                     # 信道大尺度衰落
        self._Hs = None                                     # 信道小尺度衰落
        self._H_C = None                                    # 天线域信道压缩
        self._P = 0.00012                                   # 固定发射功率分配(W)

        # 测试有干扰情况下的sinr分布情况
        # 用户数*用户数，第一个参数表明是哪个用户，第二个参数表明存在多少用户干扰
        self.sinr_interference_condition = [[[] for _ in range(self._U)] for _ in range(self._U)]

        # 用户设置参数
        Users_D = cfg.Users_D                          # 用户最大时延要求
        Users_ArriveRate = cfg.Users_ArriveRate        # 用户数据包到达率
        Users_PacketSize = cfg.Users_PacketSize        # 用户数据包大小
        Users_ps_choose = cfg.Users_ps_choose          # 用户包大小选择
        Users_Req = cfg.Users_Req                      # 用户时延满足率要求
        Users_err = cfg.Users_err                      # 用户数据块错误率
        Users_pos = cfg.Users_pos                      # 用户位置
        self._h_U = cfg.h_U                            # 用户高度
        shadow_user = np.random.normal(0, 1, size=self._U)     # User侧的shadow fading参数

        # O_RU设置参数
        self._max_P = cfg.max_P            # 每个O_RU所能承受的最大功率单元数
        self._dP = 10 ** (cfg.max_P_dBm / 10 - 3) / cfg.max_P           # 功率分配粒度(W)
        self._h_O_RU = cfg.h_O_RU              # O_RU高度
        shadow_O_RU = np.random.normal(0, 1, size=self._B)       # O_RU侧的shadow fading参数

        # self._Users = cfg["Users"]      # 用户数据
        # self._O_RUs = cfg["O_RUs"]          # O_RU数据

        self._region_bound = cfg.region_bound                # 所考虑区域大小
        self._min_distance_O_RU = cfg.min_distance_O_RU          # O_RU间最小间距
        self._min_distance_User = cfg.min_distance_User      # 用户间最小间距

        # 生成用户
        Users = []
        for i in range(self._U):
            Users.append(User(i, Users_PacketSize[Users_ps_choose], Users_D, Users_ArriveRate * 0.5,
                                    Users_Req, Users_err, self._K, shadow_user[i], Users_pos[i]))

        # 设置O_RU位置
        O_RU_pos_cfg = determine_rows_cols(self._B, self._region_bound, self._region_bound)
        # 计算行间距和列间距
        dx = self._region_bound / (O_RU_pos_cfg[1] - 1)
        dy = self._region_bound / (O_RU_pos_cfg[0] - 1)
        # 生成点的坐标
        O_RU_points = [(dx * j, dy * i) for i in range(O_RU_pos_cfg[0]) for j in range(O_RU_pos_cfg[1])]

        # 生成DFT码本
        self._DFT_codebook = 1 / np.sqrt(self._Antenna) * np.exp(
            2j * np.pi * np.outer(np.arange(self._Antenna), np.arange(self._Antenna)) / self._Antenna)

        # 生成O_RU
        O_RUs = []
        for i in range(self._B):
            # temp = {"max P": self._max_P, "location": O_RU_points[i], "shadow": shadow_O_RU[i]}
            O_RUs.append(O_RU(i, self._max_P, O_RU_points[i], self._U, self._K, self._Antenna, self._DFT_codebook))

        # 生成CPU
        self._CPU = UCDU(O_RUs, Users, self._P, self._noise * self._Bw * 1e3, self._C, self._N)

        self._slots = 0
        self._traffic = 0
        for u in self._CPU.UEs:
            self._traffic += u.packet_size * u.arrival_rate

        self._tot_bits = 0
        self._tot_Bw_counts = 0
        self.user_to_be_scheduled = []
        self.state = None

        # 设置normalizer
        if self._cfg.obs_normalize:
            self._obs_normalizer = Normalizer((self.get_obs_dim(),), clip=25).to(self._device)
        if self._cfg.reward_normalize:
            self._reward_normalizer = Normalizer((), clip=5).to(self._device)
        if self._cfg.cost_normalize:
            self._cost_normalizer = Normalizer((self.get_cost_num(),), clip=25).to(self._device)

    def reset(self):

        self._slots = 0
        self._tot_bits = 0
        self._tot_Bw_counts = 0

        # 初始化信道大尺度衰落情况
        # 暂时不考虑用户位置的变化，认为大尺度衰落特性不变
        self._Hl = np.zeros((self._K, self._U, self._B, self._Antenna))
        self._H_C = np.zeros((self._K, self._U, self._B), dtype=complex)
        for k in range(self._K):
            for u in range(self._U):
                for b in range(self._B):
                    # self._Hl[k, u, b] = ((get_dis(self._Users[u]["location"], self._O_RUs[b]["location"]) / 0.392 + 1)
                    #                ** (- 4 / 2))
                    dub = get_dis(self._CPU.UEs[u].location, self._CPU.O_RUs[b].location, self._h_U, self._h_O_RU)
                    # print(f'f{k}, user{u}, O_RU{b}, dub{dub}')
                    self._Hl[k, u, b, :] = np.sqrt(self.get_largescale(u, b, dub, self._fc))

        # 初始化信道小尺度衰落情况
        # 生成标准复高斯分布数据
        real_part = np.random.normal(0, 0.5, size=(self._K, self._U, self._B, self._Antenna))  # 实部
        imaginary_part = np.random.normal(0, 0.5, size=(self._K, self._U, self._B, self._Antenna))  # 虚部
        # 得到小尺度衰落
        self._Hs = real_part + 1j * imaginary_part
        # 得到信道矩阵
        self._H = self._Hs * self._Hl

        # 计算天线域压缩信道矩阵
        self.channel_compromize()

        self._CPU.reset(self._Hl)

        # 各个EDU整理信道信息
        for edu in self._CPU.EDUs:
            edu.collect_channels(self._H)

        # 数据包随机到达
        Queue_list = self._CPU.packets_arrival()

        # 速率估计+用户选择
        self.user_to_be_scheduled = self._CPU.user_choose()

        # 将要返回给agent的状态打包
        # 将列表转换为torch.Tensor，并扁平化为一维
        tensor_Q = torch.tensor(Queue_list).flatten()

        # 简化state，将信道的二范数作为state
        tensor_H_abs = torch.from_numpy(np.abs(self._H_C)).flatten()

        # 将时延保障速率也加入state中
        rate_list = self._CPU.get_rate_estimated()
        tensor_rate = torch.tensor(rate_list).flatten()

        # 加入受调度用户指示
        tensor_UE_indicator = torch.tensor(self._CPU.UE_scheduled_indicator).flatten()

        # self.state = torch.cat((tensor_H_abs, tensor_Q, tensor_rate), dim=0).float().to(self._device)
        # TODO: 取消CSI信息
        self.state = torch.cat((tensor_Q, tensor_rate, tensor_UE_indicator), dim=0).float().to(self._device)

        if self._cfg.obs_add_max_delay:
            tensor_m_d = torch.zeros(self._U).float().to(self._device)
            self.state = torch.cat((self.state, tensor_m_d), dim=0)

        info = {}

        state = self.state

        # TODO: obs是否需要归一化？
        if self._cfg.obs_normalize:
            origin_state = self.state
            state = self._obs_normalizer.normalize(self.state)
            info['origin_state'] = origin_state

        return state, info

    # def step(self, action: torch.Tensor):
    def step(self):

        self._slots += 1

        # 修正时延保障速率
        # rate_fix = action.tolist()
        # print(rate_fix)
        # self._CPU.rate_fix(rate_fix)

        # 下层调度
        resource_allocation, zeta = self._CPU.resource_allocation(self._Hl, self._Hs, self.user_to_be_scheduled)
        P = zeta * self._P

        # 根据调度结果更新用户数据队列，记录这一时隙内总发送比特数，并考察时延约束违反情况
        tot_bps = 0
        tot_bits = 0
        slot_Q_list = []
        slot_max_delay = []
        slot_max_delay_ratio = []
        success_users = self._U
        fail_list = []
        schedule_packets_cap = []
        user_bits = []
        traffic = []
        # TODO: 挪到CPU中！
        for i, u in enumerate(self._CPU.UEs):

            # 如果接受调度则计算传输结果，否则直接设定为0
            if i in self.user_to_be_scheduled:
                bits = self.get_rate(zeta, P, i)
                # print(f"User {i} can transmit {bits} bits")

                # 仿真数据包传输过程
                tx_bits = u.transmit(bits, self._slots)
            else:
                tx_bits = 0
            time_out_num = u.update_delay_queue()
            u.update_rate_window(tx_bits)
            u.update_delay_metric()

            # 记录各类参数
            tot_bits += tx_bits
            user_bits.append(tx_bits)
            fail_list.append(time_out_num)
            schedule_packets_cap.append(int(tx_bits / u.packet_size))
            slot_Q_list.append(u.backlog)
            max_delay = max(u.delay_queue) if u.delay_queue else 0
            slot_max_delay.append(max_delay)
            slot_max_delay_ratio.append(max_delay / u.max_delay)
            if time_out_num:
                success_users -= 1

        tot_bps = tot_bits / (self._slot_t * 1e-3)
        user_bps = [ele / (self._slot_t * 1e-3) for ele in user_bits]

        # 本时隙内总占用带宽
        # tot_Bw = 0
        Scounts = 0
        for k in range(self._K):
            if np.any(zeta[k, :, :]):
                Scounts += 1
        tot_Bw = Scounts * self._Bw * 1e3 * self._C

        # 各用户带宽使用情况
        user_Bw = []
        for u in range(self._U):
            count = 0
            for k in range(self._K):
                if np.any(zeta[k, u, :]):
                    count += 1
            user_Bw.append(count * self._Bw * 1e3 * self._C)

        self._tot_bits += tot_bits
        self._tot_Bw_counts += Scounts

        # TODO:要尽量保障时延违反率，但也要尽量降低额外加上的速率要求
        #  考虑依旧使用safe RL，但优化目标变为最小化速率修正？
        #  cost形式不变，reward变为 2 * sigmoid(-平均速率修正)
        #  reward只考虑受调度的部分用户，而不是所有用户一起考虑
        # 奖励
        # mean_rate_fix = sum(rate_fix)/self._U
        # mean_rate_fix = np.array(rate_fix) @ self._CPU.UE_scheduled_indicator / np.sum(self._CPU.UE_scheduled_indicator)
        # reward = 2 * sigmoid(-mean_rate_fix) if mean_rate_fix > 0 else 1
        reward = 0

        # 成本
        cost_list = []
        prb = []
        # 直接用超时包数和到达包数得到关系计算cost
        for i, u in enumerate(self._CPU.UEs):
            # cost_u = fail_list[i] - u.delay_req * u.arrival_rate
            cost_u = (fail_list[i] / u.arrival_rate - u.delay_req) * 3
            cost_list.append(cost_u)
            if u.totpackets > 0:
                prb.append(u.totfail / u.totpackets)
            else:
                prb.append(0)

        # 本时隙结束，下一时隙开始，随机化信道小尺度衰落情况
        # 生成标准复高斯分布数据
        real_part = np.random.normal(0, 0.5, size=(self._K, self._U, self._B, self._Antenna))  # 实部
        imaginary_part = np.random.normal(0, 0.5, size=(self._K, self._U, self._B, self._Antenna))  # 虚部
        # 得到小尺度衰落
        self._Hs = real_part + 1j * imaginary_part
        # 得到信道矩阵
        self._H = self._Hs * self._Hl

        # self.channel_compromise()

        # 清空上slot的gain计算结果
        self._CPU.clear_user_gain()

        # 各个EDU整理信道信息
        for edu in self._CPU.EDUs:
            edu.collect_channels(self._H)

        # 数据包随机到达
        Queue_list = self._CPU.packets_arrival()

        # 速率估计+用户选择
        self.user_to_be_scheduled = self._CPU.user_choose()

        # 将要返回给agent的状态打包
        # 将列表转换为torch.Tensor，并扁平化为一维
        tensor_Q = torch.tensor(Queue_list).flatten()

        # 简化state，将信道的二范数作为state
        tensor_H_abs = torch.from_numpy(np.abs(self._H_C)).flatten()

        # 将时延保障速率也加入state中
        rate_list = self._CPU.get_rate_estimated()
        tensor_rate = torch.tensor(rate_list).flatten()

        # self.state = torch.cat((tensor_H_abs, tensor_Q, tensor_rate), dim=0).float().to(self._device)
        # 加入受调度用户指示
        tensor_UE_indicator = torch.tensor(self._CPU.UE_scheduled_indicator).flatten()

        # TODO: 取消CSI信息
        self.state = torch.cat((tensor_Q, tensor_rate, tensor_UE_indicator), dim=0).float().to(self._device)

        # 添加队列最大时延与时延界的比例到obs
        if self._cfg.obs_add_max_delay:
            tensor_m_d = torch.tensor(slot_max_delay_ratio).flatten().float().to(self._device)
            self.state = torch.cat((self.state, tensor_m_d), dim=0)

        info = {
            'tot_bps': tot_bps,
            'tot_Bw': tot_Bw,
            'slot_Q_list': slot_Q_list,
            'slot_max_delay': slot_max_delay,
            'delay_break_prob': prb,
            'cost_list': cost_list,
            'user_bits': user_bits,
            'user_Bw': user_Bw,
            'user_bps': user_bps,
            'sinr_condition': self.sinr_interference_condition,
        }

        state = self.state

        # TODO: obs是否需要归一化？
        if self._cfg.obs_normalize:
            origin_state = self.state
            state = self._obs_normalizer.normalize(self.state)
            info['origin_state'] = origin_state
        else:
            info['origin_state'] = self.state

        assert not torch.isnan(state).any(), "state contains NaN values"

        reward = torch.tensor(reward).to(self._device)
        if self._cfg.reward_normalize:
            origin_reward = reward
            reward = self._reward_normalizer.normalize(reward.float())
            info['origin_reward'] = origin_reward
        else:
            info['origin_reward'] = reward

        cost = torch.tensor(cost_list).to(self._device)
        if self._cfg.cost_normalize:
            origin_cost = cost
            cost = self._cost_normalizer.normalize(cost.float())
            info['origin_cost'] = origin_cost
        else:
            info['origin_cost'] = cost

        # 返回内容：输入给agent的状态、奖励、成本、性能指标
        return state, reward, cost, info

    def save(self):

        save = {}
        if self._cfg.obs_normalize:
            save['obs_normalizer'] = self._obs_normalizer
        if self._cfg.reward_normalize:
            save['reward_normalizer'] = self._reward_normalizer
        if self._cfg.cost_normalize:
            save['cost_normalizer'] = self._cost_normalizer

        return save

    def load_norm(self, **params):

        if ('obs_norm_param' in params) and self._cfg.obs_normalize:
            self._obs_normalizer.load_state_dict(params['obs_norm_param'])
        if ('reward_norm_param' in params) and self._cfg.reward_normalize:
            self._reward_normalizer.load_state_dict(params['reward_norm_param'])
        if ('cost_norm_param' in params) and self._cfg.cost_normalize:
            self._cost_normalizer.load_state_dict(params['cost_norm_param'])

    # 计算大尺度衰落
    def get_largescale(self, u, b, dub, fc):

        beta_dB = self._d0 + self._d1 * np.log10(dub) + self._d2 * np.log10(fc * 1e-3)

        return 10 ** (- beta_dB/10)

    # TODO: 更新多天线情形
    # 计算用户在某一子带上的SINR
    def get_SINR(self, zeta, P, u, k):
        up = 0
        down = 0
        down_b = 0
        H = self._H
        # has_ICI = 0

        for v in range(H.shape[1]):

            for b in range(H.shape[2]):
                # # 暂时使用单基站MRT形式，多基站之间无协作
                # precoding_matrix = np.conj(H[k, v, b, :]) / np.linalg.norm(H[k, v, b, :])

                # TODO: 预编码现在由EDU计算，存储在O-RU处
                precoding_vector = self._CPU.O_RUs[b].user_precoding[k, v]

                # TODO: MMSE预编码式子中包含了功率分配，这里不需要额外乘
                if v == u:
                    # P0 = np.sqrt(P[k, v, b] / (self._C * self._N))
                    P0 = 1
                    H_hat = H[k, u, b, :] @ precoding_vector
                    up += np.sum(P0 * zeta[k, v, b] * H_hat)
                else:
                    # P0 = np.sqrt(P[k, v, b] / (self._C * self._N))
                    P0 = 1
                    H_hat = H[k, u, b, :] @ precoding_vector
                    down_b += np.sum(P0 * zeta[k, v, b] * H_hat)

                down += np.abs(down_b) ** 2
                down_b = 0

        sinr = np.abs(up) ** 2 / (down + self._noise * self._Bw * 1e3)

        # 用调度矩阵中非0行数量判断当前子带上是否有复用
        has_ICI = np.sum(np.any(zeta[k] != 0, axis=1))
        has_ICI = has_ICI - 1 if has_ICI != 0 else has_ICI
        if has_ICI:
            assert down > 0, 'should have ICI, but not detected'

        return sinr.real, has_ICI, np.abs(up) ** 2

    # 计算某用户在时隙内的可发送比特数
    def get_rate(self, zeta, P, u):
        first = 0
        second = 0

        for k in range(self._H.shape[0]):
            sinr, has_ICI, sinr_info = self.get_SINR(zeta, P, u, k)
            # if sinr != 0 and has_ICI:
            # if sinr != 0:
            # if sinr != 0 and sinr_info != 0:
            #     temp = 10 * np.log10(sinr_info)
            #     print(temp)
            #     self.sinr_interference_condition[u][has_ICI].append(temp)
            first += np.log2(1 + sinr)
            second += 1 - np.power((1 + sinr), -2)

        first *= self._C * self._N
        second *= self._C * self._N * (1 / np.log(2)) ** 2
        second = np.sqrt(second) * norm.ppf(1 - self._CPU.UEs[u].err_rate)

        if first - second > 0:
            return first - second
        else:
            return 0

    def channel_compromize(self):
        H = self._H

        for k in range(H.shape[0]):
            for u in range(H.shape[1]):
                for b in range(H.shape[2]):
                    # 暂时使用单基站MRT形式，多基站之间无协作
                    precoding_matrix = np.conj(H[k, u, b, :]) / np.linalg.norm(H[k, u, b, :])
                    self._H_C[k, u, b] = precoding_matrix @ H[k, u, b, :]

    # 绘制点
    def plot_points(self):

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        O_RUs_location = [O_RU.location for O_RU in self._CPU.O_RUs]
        Users_location = [User.location for User in self._CPU.UEs]

        plt.figure(figsize=(6, 6))
        plt.xlim((0, self._region_bound))
        plt.ylim((0, self._region_bound))
        plt.scatter(*zip(*O_RUs_location), color='pink', label='O_RUs')
        for i, u in enumerate(Users_location):
            plt.scatter(*u, label=f'User {i+1}')
        # plt.scatter(*zip(*Users_location), color='red', label='Users')
        plt.title('环境位置设置')
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 得到环境参数
    def get_obs_dim(self):
        if not self._cfg.obs_add_max_delay:
            # 状态空间为K*U*B的信道矩阵、各个用户的队列长度、各个用户的时延保障速率
            # 由于信道矩阵实虚部分开处理，维度要乘2(使用二范数做状态，不用乘二了)
            return self._K * self._U * self._B + self._U * 2
        else:
            # # 状态空间再加上各个用户当前队列的最大时延占时延界的比例
            # return self._K * self._U * self._B + self._U * 3
            # TODO: 状态空间取消信道信息，纯依靠CQI驱动
            #  当前为: 用户队列、用户保障速率、用户受调度情况、用户最紧迫时延
            return self._U * 4

    def get_act_dim(self):
        # if self._cfg.simple_env:
        #     # 简化功率分配和基站用户配对，只考虑各用户在子带上的调度
        #     return self._K * self._U
        # else:
        #     # 动作空间为K*U*B的调度矩阵
        #     return self._K * self._U * self._B
        # 动作变为每个用户的速率修正
        return self._U

    def get_cost_num(self):
        return self._U
