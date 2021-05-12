import sys
sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
import tightbinding.moire_tb as tbtb


n_moire = 30
n_g = 5
n_k = 60
valley = 1

(emesh, dmesh, kline) = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley, disp=True, symm=False)
n_band = emesh[0].shape[0]

fig, ax = plt.subplots()
ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
ax.set_xticklabels(["K1", "Γ", "M", "K2"])
ax.set_xlim(0, kline[-1])

# 7 bands
for i in range(4):
    plt.plot(kline, emesh[:, n_band//2+i])
    plt.plot(kline, emesh[:, n_band//2-i])

plt.plot(kline, emesh[:, n_band//2-1])
#plt.plot(kline, emesh[:, n_band//2-2])

ax.set_ylabel("Engergy (eV)")
ax.set_title("Band Structure of TBG, Nmoire: "+str(n_moire)+", Valley: "+str(valley))
ax.axvline(x=kline[0], color="black")
ax.axvline(x=kline[n_k-1], color="black")
ax.axvline(x=kline[2*n_k-1], color="black")
ax.axvline(x=kline[3*n_k-1], color="black")

plt.savefig("../fig/nsymmg2_new_7_band_n_"+str(n_moire)+"_v_"+str(valley)+".png", dpi=500)