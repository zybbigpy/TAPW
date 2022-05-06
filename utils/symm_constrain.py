import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import matplotlib.pyplot as plt
import numpy as np


def plot(atoms1, atoms2, atoms3):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    plt.scatter(atoms1[:,0], atoms1[:,1])
    plt.scatter(atoms2[:,0], atoms2[:,1])
    plt.scatter(atoms3[:,0], atoms3[:,1])


    plt.show()



c31 = tbset._set_rt_mtrx(np.pi*2/3)
c32 = tbset._set_rt_mtrx(np.pi*4/3)

moire_list = [i for i in range(65, 70, 2)]
for n_moire in moire_list:
    atoms = np.array(tbset.read_atom_pstn_list(n_moire, 'atom'))
    # conserve x-y dim
    atoms = atoms[:,:2]
    natoms = atoms.shape[0]
    print(natoms)
    inds = np.zeros([natoms, 2], dtype=int)

    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
    m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
    m_k2_vec,      m_m_vec,     rt_mtrx_half) = tbset._set_moire(n_moire)
    
    atoms_c31 = atoms@c31
    delta = 0.0001
    atoms_c31_x = np.floor(np.dot(atoms_c31, m_g_unitvec_1)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c31_y = np.floor(np.dot(atoms_c31, m_g_unitvec_2)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c31_new = atoms_c31-(atoms_c31_x)*m_unitvec_1-(atoms_c31_y)*m_unitvec_2


    atoms_c32 = atoms@c32
    atoms_c32_x = np.floor(np.dot(atoms_c32, m_g_unitvec_1)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c32_y = np.floor(np.dot(atoms_c32, m_g_unitvec_2)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_c32_new = atoms_c32-(atoms_c32_x)*m_unitvec_1-(atoms_c32_y)*m_unitvec_2
    #plot(atoms, atoms_c32, atoms_c32_new)

    atoms_l1 = atoms[:natoms//2, :]
    atoms_l2 = atoms[natoms//2:, :]
    atoms_c31_l1 = atoms_c31_new[:natoms//2, :]
    atoms_c31_l2 = atoms_c31_new[natoms//2:, :]
    assert atoms_c31_l1.shape == atoms_c31_l2.shape
    print("must be [0, 0]:", atoms_c31_l2[0])
    for i in range(natoms//2):
        dis = np.linalg.norm(atoms_c31_l1-atoms_l1[i], axis=1)
        arg = np.argmin(dis)
        inds[i][0] = arg

    for i in range(natoms//2):
        dis = np.linalg.norm(atoms_c31_l2-atoms_l2[i], axis=1)
        arg = np.argmin(dis)
        inds[i+natoms//2][0] = arg+natoms//2

    atoms_c32_l1 = atoms_c32_new[:natoms//2, :]
    atoms_c32_l2 = atoms_c32_new[natoms//2:, :]
    assert atoms_c32_l1.shape == atoms_c32_l2.shape
    print("must be [0, 0]:", atoms_c32_l2[0])
    for i in range(natoms//2):
        dis = np.linalg.norm(atoms_c32_l1-atoms_l1[i], axis=1)
        arg = np.argmin(dis)
        inds[i][1] = arg

    for i in range(natoms//2):
        dis = np.linalg.norm(atoms_c32_l2-atoms_l2[i], axis=1)
        arg = np.argmin(dis)
        inds[i+natoms//2][1] = arg+natoms//2

    for i in range(natoms):
        dis1 = np.linalg.norm(atoms[i]-atoms_c31_new[inds[i][0]])
        dis2 = np.linalg.norm(atoms[i]-atoms_c32_new[inds[i][1]])

        
        if dis1>1E-10:
            print(i, atoms[i], atoms_c31_new[inds[i][0]], atoms_c32_new[inds[i][1]])
        if dis2>1E-10:
            print(i, atoms[i], atoms_c31_new[inds[i][0]], atoms_c32_new[inds[i][1]])

    np.save("../data/group"+str(n_moire)+".npy", inds)
    print("="*50)