import sys

sys.path.append("..")

import magnetic.moire_magnetic_setup as magtbset
import tightbinding.moire_setup as tbset
import numpy as np

__author__ = 'Wangqian Miao'

n_moire  = 30
q        = 1
distance = 2.5113*magtbset.A_0

(m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magtbset._set_moire_magnetic(n_moire, q)

# test inner product between unit vec and reciprocal unit vec
print(np.dot(mm_unitvec_1, mm_g_unitvec_1)/np.pi)
print(np.dot(mm_unitvec_2, mm_g_unitvec_2)/np.pi)
print(np.dot(mm_unitvec_1, mm_g_unitvec_2)/np.pi)
print(np.dot(mm_unitvec_2, mm_g_unitvec_1)/np.pi)


a, b = magtbset.set_magnetic_atom_pstn(n_moire, 1, "../data/")
ind  = magtbset.set_magnetic_atom_neighbour_list(a, b, distance)
num_atoms = len(a)
print(num_atoms)
ind = ind%num_atoms

# print(ind[1].shape)
# print(np.sort(ind[1]))
# #print(np.sort(ind[2]))

# def washboard(x):
#     y = x-int(x/0.5)
#     return y
# r = (a[10352]-a[1])[:2]
# x = np.dot(r, m_g_unitvec_1)/(2*np.pi)
# y = np.dot(r, m_g_unitvec_2)/(2*np.pi)
# r = washboard(x)*m_unitvec_1 + washboard(y)*m_unitvec_2
# res = np.sqrt(np.dot(r,r))
# print("r", res, "distance", distance)

nlist = tbset.read_atom_neighbour_list("../data/", n_moire)
print(len(nlist))

for i in range(num_atoms):
    nlist[i].append(i)


for i in range(num_atoms):
    assert np.array_equiv(np.sort(ind[1]), np.sort(np.array(nlist[1])))==True