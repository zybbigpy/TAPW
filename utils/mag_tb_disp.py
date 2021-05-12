import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import magnetic.moire_magnetic_tb as mgtb


VALLEY_1 = 1
VALLEY_2 = -1

n_moire = 30
n_g = 5
n_k = 10
p = 1
qlist = [i for i in range(9, 10)]

for q in qlist:
    (emesh, dmesh) = mgtb.mag_tb_solver(n_moire, n_g, n_k, VALLEY_2, p, q, disp=True)
    xmax  = emesh.shape[0]
    kline = np.arange(xmax)

    n_band = emesh[0].shape[0]
    fig, ax = plt.subplots()
    ax.set_xlim(0, kline[-1])

    for i in range(q):
        plt.plot(kline, emesh[:, n_band//2+i])
        plt.plot(kline, emesh[:, n_band//2-i])
    #plt.plot(kline, emesh[:, n_band//2])
    #plt.plot(kline, emesh[:, n_band//2-1])

    ax.set_ylabel("Engergy (eV)")
    ax.set_title("Band Structure of MTBG, Nmoire: "+str(n_moire)+", Valley: -1, p = 1, q = "+str(q))
    plt.savefig("../fig/mag_band_n_"+str(n_moire)+"_v_-1_p_1_q_"+str(q)+".png", dpi=500)
    plt.clf()