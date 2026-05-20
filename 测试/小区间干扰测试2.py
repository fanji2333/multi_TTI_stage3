import numpy as np
import matplotlib.pyplot as plt


samples = 5000
Mt = 256
Mr = 4

v = np.random.normal(0, 1, size=(1, Mr))
w = np.random.normal(0, 1, size=(Mt, 1))
v /= np.linalg.norm(v)
w /= np.linalg.norm(w)
ICI = []

for sample in range(samples):
    # 生成标准复高斯分布数据
    real_part = np.random.normal(0, 0.5, size=(Mr, Mt))  # 实部
    imaginary_part = np.random.normal(0, 0.5, size=(Mr, Mt))  # 虚部
    # 得到小尺度衰落
    Hs = real_part + 1j * imaginary_part
    ICI.append(np.linalg.norm(v @ Hs @ w) ** 2)

num_bins = 100  # 直方图的柱子数量
bin_range = (0, 2)  # 数据范围
sigma2 = 0.5

# 绘制直方图
n, bins, patches = plt.hist(ICI, num_bins, bin_range, density=True, alpha=0.6, color='g')

# 绘制指数分布
x = np.linspace(0, 2, 1000)
pdf_values = 1 / sigma2 * np.exp(-1 / sigma2 * x)
plt.plot(x, pdf_values, color='r', label='Exponential PDF')

# 设置图表标签和标题
plt.xlabel('ICI')
plt.ylabel('Frequency')
plt.legend()
plt.title('ICI Frequency Distribution Histogram / random channel')

# 显示图表
plt.show()