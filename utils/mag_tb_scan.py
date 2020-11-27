import sys
sys.path.append("..")

import numpy as np
import magnetic.moire_magnetic_tb as mgtb

VALLEY_1 = 1
VALLEY_2 = -1

n_moire = 30
n_g = 5
n_k = 30
p = 1 
q = [i for i in range(2, 11)]

for qval in q:
    (emesh, dmesh) = mgtb.mag_tb_solver(n_moire, n_g, n_k, VALLEY_1, p, qval)
    emesh = np.array(emesh)
    dmesh = np.array(dmesh)
    np.save('emesh_val_1_n_'+str(n_moire)+'nk_'+str(n_k)+'ng_'+str(n_g)+'p_'+str(p)+'q_'+str(qval)+'.npy', emesh)
    np.save('dmesh_val_1_n_'+str(n_moire)+'nk_'+str(n_k)+'ng_'+str(n_g)+'p_'+str(p)+'q_'+str(qval)+'.npy', dmesh)

for qval in q:
    (emesh, dmesh) = mgtb.mag_tb_solver(n_moire, n_g, n_k, VALLEY_2, p, qval)
    emesh = np.array(emesh)
    dmesh = np.array(dmesh)
    np.save('emesh_val_2_n_'+str(n_moire)+'nk_'+str(n_k)+'ng_'+str(n_g)+'p_'+str(p)+'q_'+str(qval)+'.npy', emesh)
    np.save('dmesh_val_2_n_'+str(n_moire)+'nk_'+str(n_k)+'ng_'+str(n_g)+'p_'+str(p)+'q_'+str(qval)+'.npy', dmesh)