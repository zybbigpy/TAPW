import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import tightbinding.moire_tb as tbtb
import numpy as np

__author__ = 'Wangqian Miao'

n_moire = 31
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


# arr_a = np.sort(all_nns[1])
# arr_b = np.array(atom_neighbour_list[1])
# print("different element", np.setdiff1d(arr_a,arr_b))

for i in range(num_atoms):
    arr_a = np.sort(all_nns_mod[i])
    arr_b = np.array(atom_neighbour_list[i])
    #flag = (arr_a==arr_b).all()
    #print(arr_a.shape, arr_b.shape, "atom number:", i)

    assert(arr_a.shape==arr_b.shape)
    flag = np.array_equal(arr_a, arr_b)
    #print(flag)
    assert(flag==True)

#print((np.sort(all_nns[3450])==np.array(atom_neighbour_list[3450])).all())

(dr, dd, row, col) = tbset.set_relative_dis_ndarray(atom_pstn_list, atom_neighbour_list, m_g_unitvec_1, 
                                                    m_g_unitvec_2,  m_unitvec_1,         m_unitvec_2)
print("dr shape:", dr.shape)
print("dd shape:", dd.shape)
print(dr[1], dr[3])

(dr_new, dd_new, row_new, col_new) = tbset.set_relative_dis_ndarray_new(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
print("dr new shape:", dr_new.shape)
print("dd new shape:", dd_new.shape)
print(dr_new[1], dr_new[3])

print(row == row_new, col==col_new)


for i in range(len(col)):
    assert(col[i]==col_new[i])
# n_atom = len(atom_pstn_list)
# kmesh = tbtb._set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k)
# g_vec_list = tbtb._set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g)
# gr_mtrx, tr_mtrx = tbtb._set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
#                                         row, col, g_vec_list, atom_pstn_list, valley)
#(a, b, k) = tbtb.tightbinding_solver(n_moire, n_g, n_k, valley)