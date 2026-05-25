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
    # 1. 系统基础参数设置
    # ==========================
    M = 32  # 基站发射天线数
    K = 8  # 单天线用户数
    T_slots = 500  # 为了让CDF更有统计意义，将仿真次数提升至500
    SNR_dB = 10  # 发射端信噪比 (dB)
    rho = 10 ** (SNR_dB / 10)
    P = 1.0  # 总发射功率
    sigma2 = P / rho  # 噪声功率

    beta = M / K
    alpha = 1 / (beta * rho)  # 正则化参数

    p_alloc = (P / K) * np.ones(K)
    P_mat = np.diag(p_alloc)
    tau = 0.1 * np.ones(K)

    # ==========================
    # 2. 生成固定的信道统计信息
    # ==========================
    np.random.seed(42)
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
    # 3. 计算确等性等同 (DE) SINR
    # ==========================
    print("Calculating Deterministic Equivalent (Theorem 2)...")

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

    gamma_circ = np.zeros(K)
    for k in range(K):
        num = p_alloc[k] * (1 - tau[k] ** 2) * e[k] ** 2
        den1 = Upsilon_circ[k] * (1 - tau[k] ** 2 * (1 - (1 + e[k]) ** 2))
        den2 = (Psi_circ / rho) * (1 + e[k]) ** 2
        gamma_circ[k] = num / (den1 + den2)

    gamma_circ_dB = 10 * np.log10(gamma_circ)

    # ==========================
    # 4. 蒙特卡洛仿真获取真实 SINR
    # ==========================
    print("Simulating Empirical SINR over random realizations...")
    empirical_sinr = np.zeros((T_slots, K))

    for t in range(T_slots):
        H_true = np.zeros((K, M), dtype=np.complex128)
        H_est = np.zeros((K, M), dtype=np.complex128)

        for k in range(K):
            z_k = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
            q_k = (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)

            h_k = Theta_sqrts[k] @ z_k
            h_hat_k = Theta_sqrts[k] @ (np.sqrt(1 - tau[k] ** 2) * z_k + tau[k] * q_k)

            H_true[k, :] = h_k.conj()
            H_est[k, :] = h_hat_k.conj()

        inner_inv = np.linalg.inv(H_est @ H_est.conj().T + M * alpha * np.eye(K))
        G0 = H_est.conj().T @ inner_inv

        trace_val = np.real(np.trace(P_mat @ G0.conj().T @ G0))
        xi = np.sqrt(P / trace_val)
        G = xi * G0

        for k in range(K):
            g_k = G[:, k]
            signal_power = p_alloc[k] * np.abs(H_true[k, :] @ g_k) ** 2

            interf_power = 0
            for j in range(K):
                if j != k:
                    interf_power += p_alloc[j] * np.abs(H_true[k, :] @ G[:, j]) ** 2

            empirical_sinr[t, k] = signal_power / (interf_power + sigma2)

    empirical_sinr_dB = 10 * np.log10(empirical_sinr)

    # ==========================
    # 5. 绘制对比结果与统计分析 (新增需求)
    # ==========================
    print("Plotting results...")

    # 这里我们展示用户0和用户1的情况，如果要展示所有用户可以写成 plot_users = range(K)
    plot_users = [0, 1]

    for u in plot_users:
        # 每个用户单独一张图，包含两个子图 (1行2列)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --------------------------------------------
        # 子图 1: 真实SINR 与 确定性等同(DE) 走势对比
        # --------------------------------------------
        ax1.plot(range(T_slots), empirical_sinr_dB[:, u], marker='.', markersize=4,
                 linestyle='-', linewidth=0.8, color='#1f77b4', alpha=0.7,
                 label='Empirical SINR')
        ax1.axhline(y=gamma_circ_dB[u], color='#d62728', linestyle='--', linewidth=2.5,
                    label='Deterministic Equivalent (DE)')

        ax1.set_xlabel('Channel Realization Index (t)', fontsize=11)
        ax1.set_ylabel('SINR (dB)', fontsize=11)
        ax1.set_title(f'User {u}: Empirical vs. DE SINR', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle=':', alpha=0.7)

        # --------------------------------------------
        # 子图 2: 差值的 CDF 及 ±3dB 区域标注
        # --------------------------------------------
        # 计算差值: 真实SINR - DE SINR
        diff_dB = empirical_sinr_dB[:, u] - gamma_circ_dB[u]

        # 统计在 ±3dB 以内的概率
        prob_within_3db = np.sum(np.abs(diff_dB) <= 3) / T_slots

        # 计算 CDF 的横纵轴数据
        sorted_diff = np.sort(diff_dB)
        cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)

        # 绘制 CDF 曲线
        ax2.plot(sorted_diff, cdf, color='#2ca02c', linewidth=2, label='CDF of Difference')

        # 用半透明红色区域标注 [-3, 3] 区间
        # alpha=0.2 表示 20%的透明度
        ax2.axvspan(-3, 3, color='red', alpha=0.2, label=f'±3dB Region (Prob: {prob_within_3db:.1%})')

        ax2.set_xlabel('Difference (Empirical - DE) [dB]', fontsize=11)
        ax2.set_ylabel('CDF', fontsize=11)
        ax2.set_title(f'User {u}: CDF of Error\nProb(|Error| $\\leq$ 3dB) = {prob_within_3db:.2%}', fontsize=12)

        # 限制X轴显示范围使得重点区域更清晰
        x_limit = max(5, np.max(np.abs(sorted_diff)) + 1)
        ax2.set_xlim(-x_limit, x_limit)
        ax2.set_ylim(0, 1.05)

        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.7)

        # 调整布局并展示该用户的图表
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()