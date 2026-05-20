import matplotlib.pyplot as plt
import numpy as np


data_list = np.random.normal(0, 1, 1000)  # 这里使用正态分布生成示例数据

# 设置直方图的参数
num_bins = 30  # 直方图的柱子数量
bin_range = (min(data_list), max(data_list))  # 数据范围

# 绘制直方图
n, bins, patches = plt.hist(data_list, num_bins, bin_range, density=True, alpha=0.6, color='g')

# 设置图表标签和标题
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency Distribution Histogram')

# 显示图表
plt.show()