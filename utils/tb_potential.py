import sys
sys.path.append("..")


import numpy as np
import tightbinding.moire_tb as tbtb
import tightbinding.moire_setup as tbset
import matplotlib.pyplot as plt


for n_moire in range(45, 70, 2):
    n_g = 8
    valley = 1
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.set_aspect('equal', 'box')
    ax1.set_aspect('equal', 'box')

    ksec, glist, potential = tbtb.moire_analyze(n_moire, n_g, valley, relax=False)
    potential0 = np.abs(potential[0,0,:])
    pmax = np.max(potential0)
    print(pmax)
    potential0 = potential0/pmax
    print(potential0)

    for i in range(glist.shape[0]):
        ax0.scatter(glist[i][0], glist[i][1], marker='o', s=100, c='', edgecolors='black', alpha=0.4)
        ax0.scatter(glist[i][0], glist[i][1], s=100, c='black', alpha=np.sqrt(potential0[i]))
        ax0.set_title('Unrelaxed')


    ksec, glist, potential = tbtb.moire_analyze(n_moire, n_g, valley, relax=True)
    potential0 = np.abs(potential[0,0,:])
    pmax = np.max(potential0)
    print(pmax)
    potential0 = potential0/pmax
    print(potential0)


    for i in range(glist.shape[0]):
        ax1.scatter(glist[i][0], glist[i][1], marker='o', s=100, c='', edgecolors='black', alpha=0.4)
        ax1.scatter(glist[i][0], glist[i][1], s=100, c='black', alpha=np.sqrt(potential0[i]))
        ax1.set_title('Relaxed')
    fig.suptitle("n_moire"+str(n_moire))
    plt.tight_layout()
    plt.savefig("../output/potential_"+str(n_moire)+".png", dpi=500)
