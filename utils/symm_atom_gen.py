import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import matplotlib.pyplot as plt
import numpy as np


c31 = tbset._set_rt_mtrx(np.pi*2/3)
c32 = tbset._set_rt_mtrx(np.pi*4/3)

moire_list = [i for i in range(45, 70, 2)]
for n_moire in moire_list:
    atoms = np.array(tbset.read_atom_pstn_list(n_moire, False, True))
    print(atoms.shape)
    # conserve x-y dim
    atoms_xy = atoms[:,:2]
    natoms = atoms.shape[0]
    print("num of atoms:", natoms)
    inds = np.load("../data/group"+str(n_moire)+".npy")
    print("index shape:", inds.shape)

    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
    m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
    m_k2_vec,      m_m_vec,     rt_mtrx_half) = tbset._set_moire(n_moire)
    
    atoms_c31 = atoms_xy@c31
    delta = 0.0001
    atoms_c31_x = np.floor(np.dot(atoms_c31, m_g_unitvec_1)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c31_y = np.floor(np.dot(atoms_c31, m_g_unitvec_2)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c31_new = atoms_c31-(atoms_c31_x)*m_unitvec_1-(atoms_c31_y)*m_unitvec_2


    atoms_c32 = atoms_xy@c32
    atoms_c32_x = np.floor(np.dot(atoms_c32, m_g_unitvec_1)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c32_y = np.floor(np.dot(atoms_c32, m_g_unitvec_2)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c32_new = atoms_c32-(atoms_c32_x)*m_unitvec_1-(atoms_c32_y)*m_unitvec_2

    atoms_symm = []
    for i in range(natoms):
        indc31 = inds[i][0]
        indc32 = inds[i][1]

        #print(atoms_xy[i],atoms_c31_new[indc31],atoms_c32_new[indc32])
        new_coord_xy = (atoms_xy[i]+atoms_c31_new[indc31]+atoms_c32_new[indc32])/3
        #print(atoms[i][2],atoms[indc31][2],atoms[indc32][2])
        new_coord_z  = (atoms[i][2]+atoms[indc31][2]+atoms[indc32][2])/3
        new_coord = np.array([new_coord_xy[0], new_coord_xy[1], new_coord_z])
        atoms_symm.append(new_coord)
    
    np.savetxt("../data/relaxsymm/symmatom"+str(n_moire)+".csv", np.array(atoms_symm), header="Rx, Ry, d", delimiter=',')
    print("="*50)