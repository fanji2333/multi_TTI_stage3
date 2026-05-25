import numpy as np
import matplotlib.pyplot as plt


def generate_spatial_correlation(M, theta_bar, delta_theta=15):
    """
    生成天线域空间相关矩阵 Theta
    """
    delta_theta_rad = delta_theta / 180 * np.pi
    kai = np.pi  # 2*pi*d/lambda, 假设 d/lambda = 1/2
    R = np.zeros((M, M), dtype=np.complex128)
    for m in range(M):
        for n in range(M):
            R[m, n] = np.exp(-1j * kai * (m - n) * np.cos(theta_bar)) * \
                      np.sinc(kai * (m - n) * delta_theta_rad * np.sin(theta_bar) / np.pi)
    return R


def main():
    # ==========================
    # 1. 系统与时变信道参数设置
    # ==========================
    M = 256  # 基站发射天线数
    K = 2  # 单天线用户数
    L = 200  # 大组信道轨迹数 (蒙特卡洛组数)
    T = 40  # 每组信道轨迹内的时间步数
    SNR_dB = 10  # 发射端信噪比 (dB)
    rho_snr = 10 ** (SNR_dB / 10)
    P = 1.0  # 总发射功率
    sigma2 = P / rho_snr  # 噪声功率

    beta = M / K
    alpha = 1 / (beta * rho_snr)  # 正则化参数

    p_alloc = (P / K) * np.ones(K)
    P_mat = np.diag(p_alloc)

    # 每个用户的一阶AR模型时间相关系数 rho_k
    np.random.seed(10)
    rho_k = [0.9966] * K

    # ==========================
    # 2. 生成共用的信道统计信息
    # ==========================
    Thetas = []
    Theta_sqrts = []
    large_scale = np.random.uniform(0.3, 1.5, K)

    for k in range(K):
        theta_bar = np.random.uniform(-np.pi / 3, np.pi / 3)
        R_k = generate_spatial_correlation(M, theta_bar)
        Theta_k = large_scale[k] * R_k
        Thetas.append(Theta_k)

        vals, vecs = np.linalg.eigh(Theta_k)
        vals = np.maximum(vals, 1e-12)
        Theta_sqrts.append(vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T)

    # ==========================
    # 3. 预先计算独立于时间的 DE 基本变量 (Theorem 2)
    # ==========================
    print("Calculating Deterministic Equivalent fixed components...")

    e = np.ones(K)
    for iteration in range(200):
        e_new = np.zeros(K)
        sum_term = np.zeros((M, M), dtype=np.complex128)
        for j in range(K):
            sum_term += Thetas[j] / (1 + e[j])
        sum_term /= M
        T_mat = np.linalg.inv(sum_term + alpha * np.eye(M))
        for i in range(K):
            e_new[i] = np.real(np.trace(Thetas[i] @ T_mat) / M)
        if np.max(np.abs(e - e_new)) < 1e-6:
            e = e_new
            break
        e = e_new

    T2 = T_mat @ T_mat
    v = np.array([np.real(np.trace(Thetas[i] @ T2) / M) for i in range(K)])

    J = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            J[i, j] = np.real(np.trace(Thetas[i] @ T_mat @ Thetas[j] @ T_mat)) / (M * M * (1 + e[j]) ** 2)

    e_prime = np.linalg.inv(np.eye(K) - J) @ v

    e_k_prime = np.zeros((K, K))
    for k in range(K):
        v_k = np.array([np.real(np.trace(Thetas[i] @ T_mat @ Thetas[k] @ T_mat) / M) for i in range(K)])
        e_k_prime[k, :] = np.linalg.inv(np.eye(K) - J) @ v_k

    Psi_circ = sum([p_alloc[j] * e_prime[j] / (1 + e[j]) ** 2 for j in range(K)]) / M

    Upsilon_circ = np.zeros(K)
    for k in range(K):
        summ = sum([p_alloc[j] * e_k_prime[k, j] / (1 + e[j]) ** 2 for j in range(K) if j != k])
        Upsilon_circ[k] = summ / M

    # ==========================
    # 4. 根据信道老化时变组合出 T 个时间步的 DE SINR
    # ==========================
    gamma_circ_time = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            # AR信道老化等价映射：
            # 相关系数等价项：rho_k^t (对应原论文的 sqrt(1 - tau^2))
            # 误差方差等价项：1 - (rho_k^t)^2 (对应原论文的 tau^2)
            corr_sq = (rho_k[k] ** t) ** 2
            err_var = 1 - corr_sq

            num = p_alloc[k] * corr_sq * e[k] ** 2
            den1 = Upsilon_circ[k] * (1 - err_var * (1 - (1 + e[k]) ** 2))
            den2 = (Psi_circ / rho_snr) * (1 + e[k]) ** 2
            gamma_circ_time[t, k] = num / (den1 + den2)

    DE_SINR_dB = 10 * np.log10(gamma_circ_time)

    # ==========================
    # 5. 蒙特卡洛仿真：L 组时变轨迹
    # ==========================
    print(f"Simulating Empirical SINR over {L} trajectories, {T} steps each...")
    empirical_sinr = np.zeros((L, T, K))

    for l in range(L):
        H_true = np.zeros((K, M), dtype=np.complex128)

        # t = 0 时刻：生成初始信道并计算唯一的 Precoder G
        for k in range(K):
            z = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
            h_k = Theta_sqrts[k] @ z
            H_true[k, :] = h_k.conj()  # 行向量为共轭转置

        inner_inv = np.linalg.inv(H_true @ H_true.conj().T + M * alpha * np.eye(K))
        G0 = H_true.conj().T @ inner_inv

        trace_val = np.real(np.trace(P_mat @ G0.conj().T @ G0))
        xi = np.sqrt(P / trace_val)
        G = xi * G0  # 整个周期 T 内复用此 G

        # 遍历 T 个时间步
        for t in range(T):
            if t > 0:
                # 一阶 AR 信道演化 (针对每个用户进行)
                for k in range(K):
                    n = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
                    innovation = Theta_sqrts[k] @ n
                    # H_true 内部存的是共轭，因此 innovation 也需要共轭
                    H_true[k, :] = rho_k[k] * H_true[k, :] + np.sqrt(1 - rho_k[k] ** 2) * innovation.conj()

            # 计算当前时刻各个用户的真实 SINR
            for k in range(K):
                g_k = G[:, k]
                signal_power = p_alloc[k] * np.abs(H_true[k, :] @ g_k) ** 2

                interf_power = sum(p_alloc[j] * np.abs(H_true[k, :] @ G[:, j]) ** 2 for j in range(K) if j != k)

                empirical_sinr[l, t, k] = signal_power / (interf_power + sigma2)

    empirical_sinr_dB = 10 * np.log10(empirical_sinr)
    # 沿 L 大组方向做平均，得到长度为 T 的平均轨迹
    mean_empirical_sinr_dB = np.mean(empirical_sinr_dB, axis=0)

    # ==========================
    # 6. 绘图展示 (包含时变折线图与CDF误差分析)
    # ==========================
    print("Plotting results...")

    plot_users = [0, 1]

    for u in plot_users:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --------------------------------------------
        # 子图 1: 信道老化导致 SINR 随时间衰落的对比图
        # --------------------------------------------
        # 1. 以半透明浅色绘制 L 组原始轨迹 (alpha=0.08)
        for l in range(L):
            ax1.plot(range(T), empirical_sinr_dB[l, :, u], color='#1f77b4', alpha=0.08, linewidth=0.8)

        # 2. 绘制 L 组的均值轨迹
        ax1.plot(range(T), mean_empirical_sinr_dB[:, u], color='#1f77b4', linewidth=2.5,
                 label='Empirical Average (over L groups)')

        # 3. 绘制利用统计信息计算出的 DE SINR 时变结果
        ax1.plot(range(T), DE_SINR_dB[:, u], color='#d62728', linestyle='--', linewidth=3,
                 label='Deterministic Equivalent (DE)')

        ax1.set_xlabel('Time Step (t)', fontsize=11)
        ax1.set_ylabel('SINR (dB)', fontsize=11)
        ax1.set_title(f'User {u}: Channel Aging over Time (AR $\\rho={rho_k[u]:.3f}$)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.set_xlim([0, T - 1])

        # --------------------------------------------
        # 子图 2: 差值的 CDF 及 ±3dB 区域标注
        # --------------------------------------------
        # 计算差值: 所有 L 大组内所有 T 个时间步的真实SINR - DE SINR
        diff_dB = empirical_sinr_dB[:, :, u] - DE_SINR_dB[:, u]  # 广播减法
        diff_dB_flat = diff_dB.flatten()  # 展平为 1D 数组进行统计

        prob_within_3db = np.sum(np.abs(diff_dB_flat) <= 3) / len(diff_dB_flat)

        sorted_diff = np.sort(diff_dB_flat)
        cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)

        ax2.plot(sorted_diff, cdf, color='#2ca02c', linewidth=2, label='CDF of Difference')
        ax2.axvspan(-3, 3, color='red', alpha=0.15, label=f'±3dB Region (Prob: {prob_within_3db:.1%})')

        ax2.set_xlabel('Difference (Empirical - DE) [dB]', fontsize=11)
        ax2.set_ylabel('CDF', fontsize=11)
        ax2.set_title(
            f'User {u}: Overall Error Distribution (L×T Samples)\nProb(|Error| $\\leq$ 3dB) = {prob_within_3db:.2%}',
            fontsize=12)

        x_limit = max(5, np.max(np.abs(sorted_diff)) + 1)
        ax2.set_xlim(-x_limit, x_limit)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()