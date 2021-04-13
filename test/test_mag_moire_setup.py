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

# a, b = magtbset.set_magnetic_atom_pstn(n_moire, 1, "../data/")
# ind  = magtbset.set_magnetic_atom_neighbour_list(a, b, distance)
# num_atoms = len(a)
# print(num_atoms)
# ind = ind%num_atoms

# print(ind[1].shape)
# print(np.sort(ind[1]))

# check consistency with professor Dai's method
# def washboard(x):
#     y = x-int(x/0.5)
#     return y
# r = (a[10352]-a[1])[:2]
# x = np.dot(r, m_g_unitvec_1)/(2*np.pi)
# y = np.dot(r, m_g_unitvec_2)/(2*np.pi)
# r = washboard(x)*m_unitvec_1 + washboard(y)*m_unitvec_2
# res = np.sqrt(np.dot(r,r))
# print("r", res, "distance", distance)

# nlist = tbset.read_atom_neighbour_list("../data/", n_moire)
# print(len(nlist))

# for i in range(num_atoms):
#     print(np.array_equiv(np.sort(ind[i]), np.sort(np.array(nlist[i]))))
#     assert np.array_equiv(np.sort(ind[i]), np.sort(np.array(nlist[i])))==True

# neighbour_len_list = [subindex.shape[0] for subindex in ind]
# #print(neighbour_len_list)
# #print([subindex.shape for subindex in ind])
# atoms = np.array(a)
# k = atoms[ind[0]]
# print(k.shape)
# neighbor_array = np.concatenate(tuple(atoms[subindex] for subindex in ind))
# print(neighbor_array.shape)