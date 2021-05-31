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
q        = 1
n_g      = 5   
valley   = -1
distance = 2.5113*magtbset.A_0

(m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magtbset._set_moire_magnetic(n_moire, q)

# test inner product between unit vec and reciprocal unit vec
assert(np.dot(mm_unitvec_1, mm_g_unitvec_1)/np.pi-2<1E-10)
assert(np.dot(mm_unitvec_2, mm_g_unitvec_2)/np.pi-2<1E-10)
assert(np.dot(mm_unitvec_1, mm_g_unitvec_2)/np.pi<1E-10)
assert(np.dot(mm_unitvec_2, mm_g_unitvec_1)/np.pi<1E-10)

# test different method for glist construction
print("g vec1:", mm_g_unitvec_1)
print("q*g vec2:", mm_g_unitvec_2*q)
assert(mm_g_unitvec_1[0]/(mm_g_unitvec_1[1]*np.sqrt(3))-1<1E-10)


mm_atom_list, enlarge_mm_atom_list = magtbset.set_magnetic_atom_pstn(n_moire, q, "../data/")
atom_pstn_list = np.array(tbset.read_atom_pstn_list("../data/", n_moire))
all_nns_mag = magtbset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)



# in q==1 case should return to normal.
(m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
 m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
 m_k2_vec,      m_m_vec,     rt_mtrx_half) = tbset._set_moire(n_moire)

all_nns, enlarge_atom_pstn_list = tbset.set_atom_neighbour_list(atom_pstn_list, m_unitvec_1, m_unitvec_2)

assert(all_nns.shape == all_nns_mag.shape)

for i in range(len(mm_atom_list)):
    assert np.array_equal(all_nns[i], all_nns_mag[i])==True

print("=====finish all assertation in mag moire set up =====")