import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
import tightbinding.moire_tb as tbtb


n_moire = 30
n_g = 5
n_k = 30
valley = 1
n_band = 244

(emesh, dmesh, kline) = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley, disp=True)
assert n_band == emesh[0].shape[0]

# 7 bands
for i in range(4):
    plt.plot(kline, emesh[:, n_band//2+i])
    plt.plot(kline, emesh[:, n_band//2-i])

plt.show()