import sys
sys.path.append("..")

import numpy as np
import magnetic.moire_magnetic_setup as magset
import magnetic.periodic_guage as pg

from scipy import sparse

n_moire = 30
q = 2
B = 1/q

(m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
 mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magset._set_moire_magnetic(n_moire, q)


(mm_atom_list, enlarge_mm_atom_list) = magset.set_magnetic_atom_pstn(n_moire, q, "../data/")
n_atom = len(mm_atom_list)
ind = magset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)
(atom_pstn_2darray, atom_neighbour_2darray, row, col) = magset.set_relative_dis_ndarray(mm_atom_list, enlarge_mm_atom_list, ind)
(atom_pstn_2darray_frac, atom_neighbour_2darray_frac) = magset.set_frac_coordinate(atom_pstn_2darray, atom_neighbour_2darray, m_g_unitvec_1, m_g_unitvec_2)

num_pairs = atom_neighbour_2darray.shape[0]

x = atom_neighbour_2darray_frac[0][0]
y = atom_neighbour_2darray_frac[0][1]

print(atom_neighbour_2darray[0])
print(m_unitvec_1*x+m_unitvec_2*y)

for i in range(num_pairs):
    if (atom_neighbour_2darray[i][0]==atom_pstn_2darray[i][0]):
        print(atom_neighbour_2darray[i])

for i in range(num_pairs):
    x = atom_neighbour_2darray_frac[i][0]
    y = atom_neighbour_2darray_frac[i][1]
    
    assert np.linalg.norm(m_unitvec_1*x+m_unitvec_2*y-atom_neighbour_2darray[i][:2])<1E-10

for i in range(num_pairs):
    x = atom_pstn_2darray_frac[i][0]
    y = atom_pstn_2darray_frac[i][1]

    assert np.linalg.norm(m_unitvec_1*x+m_unitvec_2*y-atom_pstn_2darray[i][:2])<1E-10

ab_phase = pg.set_ab_phase_list(num_pairs, B, atom_pstn_2darray_frac, atom_neighbour_2darray_frac)
ab_phase = np.exp(2j*np.pi*ab_phase)
abphase_mtrx = sparse.csr_matrix((ab_phase, (row, col)), shape=(n_atom, n_atom))  
print(abphase_mtrx.shape)

abphase_mtrx_cc = (abphase_mtrx.transpose()).conjugate()
abphase_mtrx_delta = abphase_mtrx - abphase_mtrx_cc

print(abphase_mtrx_delta.max())
if abphase_mtrx_delta.max()>1.0E-10:
    print(abphase_mtrx_delta.max())
    raise Exception("abphase matrix is not hermitian?!") 