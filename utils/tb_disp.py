import sys
sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
import tightbinding.moire_tb as tbtb


n_moire = 30
n_g = 5
n_k = 60
valley = 1

(emesh, dmesh, kline) = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley, disp=True, symm=True)
n_band = emesh[0].shape[0]

fig, ax = plt.subplots()
ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
ax.set_xlim(0, kline[-1])

# 9 bands
for i in range(1):
    plt.plot(kline, emesh[:, n_band//2+i],'-b')
    plt.plot(kline, emesh[:, n_band//2-i],'-b')

plt.plot(kline, emesh[:, n_band//2-1],'-b')
#plt.plot(kline, emesh[:, n_band//2-2])

(emesh, dmesh, kline) = tbtb.tightbinding_solver(n_moire, n_g, n_k, -valley, disp=True, symm=True)

for i in range(1):
    plt.plot(kline, emesh[:, n_band//2+i],'--r')
    plt.plot(kline, emesh[:, n_band//2-i],'--r')

plt.plot(kline, emesh[:, n_band//2-1],'--r')
#plt.plot(kline, emesh[:, n_band//2-2])

ax.set_ylabel("Engergy (eV)")
ax.set_title("TB Model, Flat Bands of TBG ")
ax.axvline(x=kline[0], color="black")
ax.axvline(x=kline[n_k-1], color="black")
ax.axvline(x=kline[2*n_k-1], color="black")
ax.axvline(x=kline[3*n_k-1], color="black")

plt.savefig("../fig/flatbands"+str(n_moire)+"_v_"+str(valley)+".png", dpi=500)