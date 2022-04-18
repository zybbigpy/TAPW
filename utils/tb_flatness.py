import sys
sys.path.append("..")


import numpy as np
import tightbinding.moire_tb as tbtb


n_g = 7
n_k = 10

def cal_flatness(emesh, nmoire, valley):
    nband = emesh[0].shape[0]
    e1 = emesh[:, nband//2]
    e2 = emesh[:, nband//2-1]
    e  = []
    print(e1.shape, e2.shape)
    for i in range(e1.shape[0]):
        e.append(e1[i])
    for i in range(e2.shape[0]):
        e.append(e2[i])
    
    print("moire:", nmoire, "valley:", valley, "flatness:", np.var(np.array(e)))




for n_moire in range(55, 70, 2):

    emeshvp1, _, _ = tbtb.tightbinding_solver(n_moire, n_g, n_k,  1, False, True, True)
    print(emeshvp1.shape)
    cal_flatness(emeshvp1, n_moire, 1)

    emeshvn1, _, _ = tbtb.tightbinding_solver(n_moire, n_g, n_k, -1, False, True, True)
    print(emeshvn1.shape)
    cal_flatness(emeshvn1, n_moire, -1)


    