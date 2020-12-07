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

(emesh, dmesh) = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley, disp=True)
assert n_band == emesh[0].shape[0]

# x is kline
x = np.arange(emesh.shape[0])
print(x)
fig, ax = plt.subplots()
ax.set_xticks([0, 30, 60, 90])
ax.set_xticklabels(["K1", "Γ", "M", "K2"])
ax.set_xlim(0, max(x))

# 7 bands
for i in range(4):
    plt.plot(x, emesh[:, n_band//2+i])
    plt.plot(x, emesh[:, n_band//2-i])

ax.set_ylabel("Engergy (eV)")
ax.set_title("Band Structure of TBG, Nmoire: "+str(n_moire)+", Valley: "+str(valley))
ax.axvline(x=0, color="black")
ax.axvline(x=30, color="black")
ax.axvline(x=60, color="black")
ax.axvline(x=90, color="black")

plt.savefig("../fig/band_n_"+str(n_moire)+"_v_"+str(valley)+".png", dpi=500)