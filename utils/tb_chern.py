import sys
sys.path.append("..")

import numpy as np
import tightbinding.moire_tb as tbtb
import tightbinding.moire_chern as tbchern

n_moire = 30
n_g = 5
n_k = 18
valley = 1
# Control band plotted

emesh, dmesh, _, transmatlist, neighbor_map = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley)

np.save("bands.npy", dmesh)
np.save("trans.npy", np.array(transmatlist))
np.save("map.npy", np.array(neighbor_map))

# dmesh = np.load("D20-N30-nG5Nk18Valley1.npy")
# trans = np.load("delm.npy")
# nmap  = np.load("mplusone.npy")
dmesh = np.load("bands.npy")
trans = np.load("trans.npy")
nmap = np.load("map.npy")

print(dmesh.shape, trans.shape, nmap.shape)
nband = dmesh.shape[2]
dmesh = dmesh[:, :, (nband // 2-3):(nband // 2+3)]
for i in range(6):
    print(tbchern.cal_chern(dmesh, n_k, i, i, trans, nmap))
