from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt


T = 160
fc = 6.7    # GHz
c = 3e8
v = 3/3.6   # m/s

fd = 6.7e9 * v / c * 0.9
tti = np.linspace(0, T-1, T)
dt = 0.5e-3    # s

corr_bassel = jv(0, 2 * np.pi * fd * tti * dt)

data = corr_bassel ** 2 * 0.8 + 0.2

plt.plot(tti, data)
plt.grid(True)
plt.show()

plt.figure()
plt.plot(tti, 10 * np.log10(data))
plt.grid(True)
plt.show()