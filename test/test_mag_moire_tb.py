import sys
sys.path.append("..")

import magnetic.moire_magnetic_tb as magtb

import numpy as np
import matplotlib.pyplot as plt

VALLEY_1 = 1
VALLEY_2 = -1
n_moire = 30
n_g = 5
n_k = 10
p   = 0
q   = 5

# emesh = np.load("/Users/wqmiao/Workspace/TBG/data/emesh_n30_p1_q6_v-1.npy")

emesh, dmesh = magtb.mag_tb_solver_periodic(n_moire, n_g, n_k, VALLEY_1, p, q, disp=True)
xmax  = emesh.shape[0]
kline = np.arange(xmax)

n_band = emesh[0].shape[0]
ind1 = n_band//2-q-2
ind2 = n_band//2+q+2

fig, ax = plt.subplots()
ax.set_xlim(0, kline[-1])

num = 0
for band_num in range(ind1, ind2):
    plt.plot(kline, emesh[:, band_num])
    print("band index", band_num)

print(num)
plt.show()
