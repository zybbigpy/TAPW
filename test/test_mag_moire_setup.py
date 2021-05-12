import sys

from sklearn import neighbors

sys.path.append("..")

import magnetic.moire_magnetic_setup as magtbset
import tightbinding.moire_setup as tbset
import magnetic.moire_magnetic_tb as magtb

import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Wangqian Miao'


n_moire  = 30
q        = 3
n_g      = 5   
valley   = -1
distance = 2.5113*magtbset.A_0

(m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magtbset._set_moire_magnetic(n_moire, q)

# test inner product between unit vec and reciprocal unit vec
print(np.dot(mm_unitvec_1, mm_g_unitvec_1)/np.pi)
print(np.dot(mm_unitvec_2, mm_g_unitvec_2)/np.pi)
print(np.dot(mm_unitvec_1, mm_g_unitvec_2)/np.pi)
print(np.dot(mm_unitvec_2, mm_g_unitvec_1)/np.pi)

# test different method for glist construction
print("g vec1:", mm_g_unitvec_1)
print("q*g vec2:", mm_g_unitvec_2*q)
print("check degree:(should be 1)", mm_g_unitvec_1[0]/(mm_g_unitvec_1[1]*np.sqrt(3)))

g_vec_list = magtb._set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley, q)
g_vec_list_1 = magtb._set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley, 1)

print(g_vec_list_1.shape)
print(g_vec_list.shape)
print(g_vec_list[:,0].shape)
plt.scatter(g_vec_list[:, 0], g_vec_list[:, 1])
plt.scatter(g_vec_list_1[:, 0], g_vec_list_1[:, 1], c='red', marker='v')
plt.savefig("../fig/glist_mtb_v_"+str(valley)+".png", dpi=600)
plt.cla()


mm_atom_list, enlarge_mm_atom_list = magtbset.set_magnetic_atom_pstn(n_moire, q, "../data/")
atom_pstn_list = np.array(tbset.read_atom_pstn_list("../data/", n_moire))
plt.scatter(enlarge_mm_atom_list[:,0], enlarge_mm_atom_list[:,1])
plt.scatter(mm_atom_list[:,0], mm_atom_list[:,1],c='red')
plt.scatter(atom_pstn_list[:,0], atom_pstn_list[:,1], c='green')
plt.savefig("../fig/atoms_mag.png", dpi=600)

print("non mag atom numbers:", len(mm_atom_list)/q)
print("mm atom numbers:", len(mm_atom_list))
print("enlarge mm atom numbers:", len(enlarge_mm_atom_list))

all_nns_mag = magtbset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)



# # in q==1 case should return to normal.
# (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
#  m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
#  m_k2_vec,      m_m_vec,     rt_mtrx_half) = tbset._set_moire(n_moire)

# all_nns, enlarge_atom_pstn_list = tbset.set_atom_neighbour_list(atom_pstn_list, m_unitvec_1, m_unitvec_2)

# print(all_nns.shape)
# print(all_nns_mag.shape)

# for i in range(len(mm_atom_list)):
#     assert np.array_equal(all_nns[i], all_nns_mag[i])==True