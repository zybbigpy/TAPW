import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import tightbinding.moire_tb as tbtb

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Wangqian Miao'

n_moire = 30
n_k     = 3
n_g     = 5
valley  = -1


print("="*100)
# test innder product between unit vec and reciprocal vec
assert(np.dot(tbset.A_UNITVEC_1, tbset.A_G_UNITVEC_1)/np.pi-2<1E-10)
assert(np.dot(tbset.A_UNITVEC_2, tbset.A_G_UNITVEC_2)/np.pi-2<1E-10)

# test info log
tbset.system_info_log(n_moire)

# test atom pstn list construction
atoms = tbset.set_atom_pstn_list(n_moire)
tbset.save_atom_pstn_list(atoms, "./", n_moire)

# test load neighour list, atom position list
atom_neighbour_list = tbset.read_atom_neighbour_list("../data/", n_moire)
atom_pstn_list = tbset.read_atom_pstn_list(n_moire, 'corrugation')
num_atoms = len(atom_pstn_list)
print("num of atoms:", num_atoms)

# test read file result
print(atom_neighbour_list[2567][7])
print(atom_pstn_list[267] == atoms[267])
print(atom_pstn_list[1249] == atoms[1249])


(m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
 m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
 m_k2_vec,      m_m_vec,     rt_mtrx_half) = tbset._set_moire(n_moire)

print(np.dot(m_unitvec_1, m_g_unitvec_1)/np.pi)
print(np.dot(m_unitvec_2, m_g_unitvec_2)/np.pi)

all_nns, enlarge_atom_pstn_list = tbset.set_atom_neighbour_list(atom_pstn_list, m_unitvec_1, m_unitvec_2)

all_nns_mod = all_nns%num_atoms
print("neighour index shape:", all_nns.shape)

# compare the result of brute force search and KDTree search
for i in range(num_atoms):
    arr_a = np.sort(all_nns_mod[i])
    arr_b = np.array(atom_neighbour_list[i])

    assert(arr_a.shape==arr_b.shape)
    flag = np.array_equal(arr_a, arr_b)
    assert(flag==True)

(dr, dd, row, col) = tbset.set_relative_dis_ndarray(atom_pstn_list, atom_neighbour_list, m_g_unitvec_1, 
                                                    m_g_unitvec_2,  m_unitvec_1,         m_unitvec_2)

print(dr.shape, dd.shape)
print("finish assertation in moire tb set up")
print("="*100)