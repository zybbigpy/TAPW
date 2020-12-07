import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import tightbinding.moire_tb as tbtb
import numpy as np

__author__ = 'Wangqian Miao'

n_moire = 30
n_k     = 3
n_g     = 5
valley  = 1

# test innder product between unit vec and reciprocal vec
print(np.dot(tbset.A_UNITVEC_1, tbset.A_G_UNITVEC_1)/np.pi)
print(np.dot(tbset.A_UNITVEC_2, tbset.A_G_UNITVEC_2)/np.pi)

# test infor log
tbset.system_info_log(n_moire)

# test atom pstn list construction
atoms = tbset.set_atom_pstn_list(n_moire)
tbset.save_atom_pstn_list(atoms, "../data/", n_moire)

# test load neighour list, atom position list
atom_neighbour_list = tbset.read_atom_neighbour_list("../data/", n_moire)
atom_pstn_list = tbset.read_atom_pstn_list("../data/", n_moire)

# test read file result
print(atom_neighbour_list[2567][7])
print(atom_pstn_list[267] == atoms[267])
print(atom_pstn_list[1249] == atoms[1249])


(m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
 m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
 m_k2_vec,      m_m_vec,     rt_mtrx_half) = tbset._set_moire(n_moire)

print(np.dot(m_unitvec_1, m_g_unitvec_1)/np.pi)
print(np.dot(m_unitvec_2, m_g_unitvec_2)/np.pi)

(dr, dd, row, col) = tbset.set_relative_dis_ndarray(atom_pstn_list, atom_neighbour_list, m_g_unitvec_1, 
                                                    m_g_unitvec_2,  m_unitvec_1,         m_unitvec_2)
print("dr shape:", dr.shape)
print("dd shape:", dd.shape)

# n_atom = len(atom_pstn_list)
# kmesh = tbtb._set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k)
# g_vec_list = tbtb._set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g)
# gr_mtrx, tr_mtrx = tbtb._set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
#                                         row, col, g_vec_list, atom_pstn_list, valley)
(a, b, k) = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley)