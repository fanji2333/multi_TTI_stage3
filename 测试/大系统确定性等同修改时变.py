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
    Mt = 256  # 基站发射天线数
    Mr = 4  # 用户接收天线数 (扩展为 4 天线)
    K = 4  # 单天线用户数
    L = 20000  # 大组信道轨迹数 (蒙特卡洛组数)
    T = 160  # 每组信道轨迹内的时间步数
    SNR_dB = 10  # 发射端信噪比 (dB)
    rho_snr = 10 ** (SNR_dB / 10)
    P = 1.0  # 总发射功率
    sigma2 = P / rho_snr  # 噪声功率

    beta = Mt / K
    alpha = 1 / (beta * rho_snr)  # 正则化参数

    p_alloc = (P / K) * np.ones(K)
    P_mat = np.diag(p_alloc)

    # 每个用户的一阶AR模型时间相关系数 rho_k
    np.random.seed(42)
    rho_k = [0.9966] * K

    # ==========================
    # 2. 生成共用的信道统计信息
    # ==========================
    Thetas_eff = []
    R_Tk_list = []  # 保存发射端相关矩阵，供启发式估计算法使用
    lambda_max_R_list = []  # 保存接收端组合器增益，供启发式估计算法使用

    R_Tk_sqrts = []
    R_Rk_sqrts = []

    large_scale = np.random.uniform(0.3, 1.5, K)

    for k in range(K):
        theta_bar = np.random.uniform(-np.pi / 3, np.pi / 3)
        R_Tk = generate_spatial_correlation(Mt, theta_bar)
        R_Tk_list.append(R_Tk)

        R_Rk = generate_spatial_correlation(Mr, theta_bar)

        vals_R, vecs_R = np.linalg.eigh(R_Rk)
        lambda_max_R = np.max(vals_R)
        lambda_max_R_list.append(lambda_max_R)

        Theta_eff_k = large_scale[k] * lambda_max_R * R_Tk
        Thetas_eff.append(Theta_eff_k)

        vals_T, vecs_T = np.linalg.eigh(R_Tk)
        R_Tk_sqrts.append(vecs_T @ np.diag(np.sqrt(np.maximum(vals_T, 0))) @ vecs_T.conj().T)

        R_Rk_sqrts.append(vecs_R @ np.diag(np.sqrt(np.maximum(vals_R, 0))) @ vecs_R.conj().T)

    # ==========================
    # 3. 预先计算独立于时间的 DE 基本变量 (使用等效协方差矩阵)
    # ==========================
    print("Calculating Deterministic Equivalent fixed components using Effective Thetas...")

    e = np.ones(K)
    for iteration in range(200):
        e_new = np.zeros(K)
        sum_term = np.zeros((Mt, Mt), dtype=np.complex128)
        for j in range(K):
            sum_term += Thetas_eff[j] / (1 + e[j])
        sum_term /= Mt
        T_mat = np.linalg.inv(sum_term + alpha * np.eye(Mt))
        for i in range(K):
            e_new[i] = np.real(np.trace(Thetas_eff[i] @ T_mat) / Mt)
        if np.max(np.abs(e - e_new)) < 1e-6:
            e = e_new
            break
        e = e_new

    T2 = T_mat @ T_mat
    v = np.array([np.real(np.trace(Thetas_eff[i] @ T2) / Mt) for i in range(K)])

    J = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            J[i, j] = np.real(np.trace(Thetas_eff[i] @ T_mat @ Thetas_eff[j] @ T_mat)) / (Mt * Mt * (1 + e[j]) ** 2)

    e_prime = np.linalg.inv(np.eye(K) - J) @ v

    e_k_prime = np.zeros((K, K))
    for k in range(K):
        v_k = np.array([np.real(np.trace(Thetas_eff[i] @ T_mat @ Thetas_eff[k] @ T_mat) / Mt) for i in range(K)])
        e_k_prime[k, :] = np.linalg.inv(np.eye(K) - J) @ v_k

    Psi_circ = sum([p_alloc[j] * e_prime[j] / (1 + e[j]) ** 2 for j in range(K)]) / Mt

    Upsilon_circ = np.zeros(K)
    for k in range(K):
        summ = sum([p_alloc[j] * e_k_prime[k, j] / (1 + e[j]) ** 2 for j in range(K) if j != k])
        Upsilon_circ[k] = summ / Mt

    gamma_circ_time = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            corr_sq = (rho_k[k] ** t) ** 2
            err_var = 1 - corr_sq

            num = p_alloc[k] * corr_sq * e[k] ** 2
            den1 = Upsilon_circ[k] * (1 - err_var * (1 - (1 + e[k]) ** 2))
            den2 = (Psi_circ / rho_snr) * (1 + e[k]) ** 2
            gamma_circ_time[t, k] = num / (den1 + den2)

    DE_SINR_dB = 10 * np.log10(gamma_circ_time)

    # ==========================
    # 4. 新增: 计算上传代码中的 Heuristic SINR 估算
    # ==========================
    print("Calculating Heuristic SINR Estimate from snippet...")
    Heuristic_SINR_time = np.zeros((T, K))
    for k in range(K):
        # 预先计算独立于时间的干扰项 (对应 snippet 中的 interference)
        interf_k = 0
        for j in range(K):
            if j != k:
                interf_k += p_alloc[j] * lambda_max_R_list[k] * np.trace(R_Tk_list[j] @ R_Tk_list[k]).real / Mt

        for t in range(T):
            aging_factor = rho_k[k] ** (2 * t)
            # 对应 gain
            gain = aging_factor * large_scale[k] * p_alloc[k] * lambda_max_R_list[k] * Mt
            # 对应 mu_loss
            mu_loss = (1 - aging_factor) * large_scale[k] * interf_k

            # 对应 sinr_estimate_list.append
            Heuristic_SINR_time[t, k] = gain / (mu_loss + sigma2)

    Est_SINR_dB = 10 * np.log10(Heuristic_SINR_time)

    # ==========================
    # 5. 蒙特卡洛仿真：L 组 MIMO 时变轨迹
    # ==========================
    print(f"Simulating Empirical MIMO EZF SINR over {L} trajectories, {T} steps each...")
    empirical_sinr = np.zeros((L, T, K))

    for l in range(L):
        # 存储当前大组各用户在各时刻的真实 MIMO 信道矩阵 (N_r x M)
        H_true_t = []

        H_eff_0 = np.zeros((K, Mt), dtype=np.complex128)
        W_combiners = []  # 存储过时的 combiners

        # t = 0 时刻：生成初始 MIMO 信道，计算 combiner 并降维为等效信道计算 Precoder
        for k in range(K):
            Z = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
            # MIMO Kronecker 信道模型
            H_k = np.sqrt(large_scale[k]) * (R_Rk_sqrts[k] @ Z @ R_Tk_sqrts[k])
            H_true_t.append(H_k)

            # SVD 分解提取 combiner
            U, S, Vh = np.linalg.svd(H_k, full_matrices=False)
            w_k = U[:, 0]  # 取最大的左奇异向量，shape (N_r,)
            W_combiners.append(w_k)

            # 降维为等效 MISO 信道，并存入行矩阵
            h_eff_k = w_k.conj() @ H_k  # shape (M,)
            H_eff_0[k, :] = h_eff_k

        # 基于等效信道 H_eff_0 进行 RZF 预编码设计
        inner_inv = np.linalg.inv(H_eff_0 @ H_eff_0.conj().T + Mt * alpha * np.eye(K))
        G0 = H_eff_0.conj().T @ inner_inv

        trace_val = np.real(np.trace(P_mat @ G0.conj().T @ G0))
        xi = np.sqrt(P / trace_val)
        G = xi * G0  # 整个周期 T 内复用此 G 和 W_combiners

        # 遍历 T 个时间步
        for t in range(T):
            if t > 0:
                # 信道时变老化
                for k in range(K):
                    Z_innov = (np.random.randn(Mr, Mt) + 1j * np.random.randn(Mr, Mt)) / np.sqrt(2)
                    H_innov = np.sqrt(large_scale[k]) * (R_Rk_sqrts[k] @ Z_innov @ R_Tk_sqrts[k])
                    H_true_t[k] = rho_k[k] * H_true_t[k] + np.sqrt(1 - rho_k[k] ** 2) * H_innov

            # 计算当前时刻各用户的真实 SINR
            for k in range(K):
                # 结合过时的 combiner 计算瞬时的等效信道接收
                h_eff_k_t = W_combiners[k].conj() @ H_true_t[k]  # shape (M,)

                g_k = G[:, k]
                signal_power = p_alloc[k] * np.abs(h_eff_k_t @ g_k) ** 2

                interf_power = 0
                for j in range(K):
                    if j != k:
                        interf_power += p_alloc[j] * np.abs(h_eff_k_t @ G[:, j]) ** 2

                empirical_sinr[l, t, k] = signal_power / (interf_power + sigma2)

    empirical_sinr_dB = 10 * np.log10(empirical_sinr)
    mean_empirical_sinr_dB = np.mean(empirical_sinr_dB, axis=0)

    # ==========================
    # 6. 绘图展示
    # ==========================
    print("Plotting results...")
    plot_users = range(K)

    for u in plot_users:
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]

        emp_u = empirical_sinr_dB[:, :, u]
        de_u = DE_SINR_dB[:, u]
        est_u = Est_SINR_dB[:, u]  # 提取启发式估计算法轨迹
        mean_u = mean_empirical_sinr_dB[:, u]

        # --------------------------------------------
        # 子图 1: 添加启发式 SINR 估计算法对比
        # --------------------------------------------
        for l in range(L):
            ax1.plot(range(T), emp_u[l, :], color='#1f77b4', alpha=0.08, linewidth=0.8)

        ax1.plot(range(T), mean_u, color='r', linewidth=2.5, label='Empirical Mean')
        ax1.plot(range(T), de_u, color='#d62728', linestyle='--', linewidth=3, label='DE SINR')
        # 新增绘制 Heuristic 估计算法
        ax1.plot(range(T), est_u, color='#ff7f0e', linestyle='-.', linewidth=2.5, label='Heuristic Est SINR')

        ax1.set_xlabel('Time Step (t)', fontsize=11)
        ax1.set_ylabel('SINR (dB)', fontsize=11)
        ax1.set_title(f'User {u}: SINR Aging over Time ($\\rho={rho_k[u]:.3f}$)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.set_xlim([0, T - 1])

        # --------------------------------------------
        # 子图 2: 不作改变
        # --------------------------------------------
        diff_dB_flat = (emp_u - de_u).flatten()
        prob_within_3db = np.sum(np.abs(diff_dB_flat) <= 3) / len(diff_dB_flat)

        sorted_diff = np.sort(diff_dB_flat)
        cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)

        ax2.plot(sorted_diff, cdf, color='#2ca02c', linewidth=2, label='CDF of Error')
        ax2.axvspan(-3, 3, color='red', alpha=0.15, label=f'±3dB (Prob: {prob_within_3db:.1%})')

        ax2.set_xlabel('Error (Empirical - DE) [dB]', fontsize=11)
        ax2.set_ylabel('CDF', fontsize=11)
        ax2.set_title(f'User {u}: Overall Error Distribution (DE vs Emp)', fontsize=12)

        x_limit = max(5, np.max(np.abs(sorted_diff)) + 1)
        ax2.set_xlim(-x_limit, x_limit)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle=':', alpha=0.7)

        # --------------------------------------------
        # 子图 3: 新增关于 Heuristic 估计算法的误差频率追踪
        # --------------------------------------------
        prob_de_3db_t = np.sum(np.abs(emp_u - de_u) <= 3, axis=0) / L
        prob_mean_3db_t = np.sum(np.abs(emp_u - mean_u) <= 3, axis=0) / L
        prob_est_3db_t = np.sum(np.abs(emp_u - est_u) <= 3, axis=0) / L  # 新增

        ax3.plot(range(T), prob_de_3db_t, color='#d62728', marker='o', markersize=4, label='P(|Emp - DE| <= 3dB)')
        ax3.plot(range(T), prob_est_3db_t, color='#ff7f0e', marker='d', markersize=4,
                 label='P(|Emp - Heuristic| <= 3dB)')
        ax3.plot(range(T), prob_mean_3db_t, color='#1f77b4', marker='s', markersize=4,
                 label='P(|Emp - Mean| <= 3dB)')

        ax3.set_xlabel('Time Step (t)', fontsize=11)
        ax3.set_ylabel('Probability', fontsize=11)
        ax3.set_title(f'User {u}: Probability of Absolute Error <= 3dB over Time', fontsize=12)
        ax3.set_ylim(0, 1.05)
        ax3.set_xlim([0, T - 1])
        ax3.legend(loc='lower left')
        ax3.grid(True, linestyle=':', alpha=0.7)

        # --------------------------------------------
        # 子图 4: 不作改变
        # --------------------------------------------
        diff_de_mean = de_u - mean_u

        ax4.plot(range(T), diff_de_mean, color='#9467bd', linewidth=2, marker='^', markersize=5)
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)

        ax4.set_xlabel('Time Step (t)', fontsize=11)
        ax4.set_ylabel('Difference (dB)', fontsize=11)
        ax4.set_title(f'User {u}: DE-SINR minus Mean SINR over Time', fontsize=12)
        ax4.set_xlim([0, T - 1])

        max_diff = np.max(np.abs(diff_de_mean))
        y_lim = max(0.5, max_diff * 1.5)
        ax4.set_ylim(-y_lim, y_lim)
        ax4.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()