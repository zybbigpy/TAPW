import sys
sys.path.append("..")

import numpy as np
import magnetic.moire_magnetic_tb as mgtb


VALLEY_1 = 1
VALLEY_2 = -1

n_moire = 30
n_g = 5
n_k = 10
qlist = [i for i in range(2, 11)]


# p = 1
# for q in qlist:
#     (emesh, dmesh) = mgtb.mag_tb_solver(n_moire, n_g, n_k, VALLEY_2, p, q, disp=True)
#     print("(p, q)", (p,q), emesh.shape, dmesh.shape)
#     np.save("../data/emesh_n30_"+"p"+str(p)+"_q"+str(q)+"_v"+str(VALLEY_2)+".npy", emesh)
#     np.save("../data/dmesh_n30_"+"p"+str(p)+"_q"+str(q)+"_v"+str(VALLEY_2)+".npy", dmesh)


p = 1
for q in qlist:
    (emesh, dmesh) = mgtb.mag_tb_solver(n_moire, n_g, n_k, VALLEY_2, p, q, disp=True)
    print("(p, q)", (p,q), emesh.shape, dmesh.shape)
    np.save("../data/emesh_n30_"+"p"+str(p)+"_q"+str(q)+"_v"+str(VALLEY_2)+".npy", emesh)
    np.save("../data/dmesh_n30_"+"p"+str(p)+"_q"+str(q)+"_v"+str(VALLEY_2)+".npy", dmesh)
