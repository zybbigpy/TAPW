import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import matplotlib.pyplot as plt
import numpy as np


def plot(atoms1, atoms2):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    plt.scatter(atoms1[:, 0], atoms1[:, 1])
    plt.scatter(atoms2[:, 0], atoms2[:, 1])
    plt.show()


c31 = tbset._set_rt_mtrx(np.pi*2/3)
c32 = tbset._set_rt_mtrx(np.pi*4/3)
c6 = tbset._set_rt_mtrx(np.pi/3)
c2x = np.array([[-1, 0], [0, 1]])


def symm_c2x(atoms):

    return atoms@c2x@c6


def symm_c31(atoms):

    return atoms@c31


def symm_c32(atoms):

    return atoms@c32


def symm_c2xc31(atoms):

    return atoms@c2x@c6@c31


def symm_c2xc32(atoms):
    return atoms@c2x@c6@c32


def symm_reconstruct(m_g_unitvec1, m_g_unitvec2, symm, atoms, halfatoms):
    # apply symmetry
    atoms_symm = symm(atoms)
    delta = 0.0001
    # x-y coordinate
    atoms_symm_x = np.floor(np.dot(atoms_symm, m_g_unitvec_1)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_symm_y = np.floor(np.dot(atoms_symm, m_g_unitvec_2)/(np.pi*2)+delta).reshape(natoms, 1)
    # move to the original unit cell
    atoms_symm_new = atoms_symm-(atoms_symm_x)*m_unitvec_1-(atoms_symm_y)*m_unitvec_2

    atoms_symm_l1 = atoms_symm_new[:halfatoms, :]
    atoms_symm_l2 = atoms_symm_new[halfatoms:, :]

    return atoms_symm_new, atoms_symm_l1, atoms_symm_l2


def find_ind(atoms_new, atoms_old, halfatoms, flag1, flag2, inds, ind):
    assert atoms_new.shape == atoms_old.shape
    for i in range(halfatoms):
        dis = np.linalg.norm(atoms_new-atoms_old[i], axis=1)
        arg = np.argmin(dis)
        inds[i+flag1*halfatoms][ind] = arg+flag2*halfatoms


moire_list = [i for i in range(30, 70)]
for n_moire in moire_list:
    atoms = np.array(tbset.read_atom_pstn_list(n_moire, 'atom'))
    # conserve x-y dim
    atoms = atoms[:, :2]
    natoms = atoms.shape[0]
    halfatoms = natoms // 2
    print("number of total atoms:", natoms)

    inds = np.zeros([natoms, 5], dtype=int)
    atoms_l1 = atoms[:natoms // 2, :]
    atoms_l2 = atoms[natoms // 2:, :]
    (m_unitvec_1, m_unitvec_2, m_g_unitvec_1, m_g_unitvec_2, m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec,
     rt_mtrx_half) = tbset._set_moire(n_moire)

    atoms_c31, atoms_c31_l1, atoms_c31_l2 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c31, atoms, halfatoms)
    atoms_c32, atoms_c32_l1, atoms_c32_l2 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c32, atoms, halfatoms)
    atoms_c2x, atoms_c2x_l1, atoms_c2x_l2 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c2x, atoms, halfatoms)
    #plot(atoms, atoms_c2x)
    atoms_c2xc31, atoms_c2xc31_l1, atoms_c2xc31_l2 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c2xc31, atoms,
                                                                      halfatoms)
    atoms_c2xc32, atoms_c2xc32_l1, atoms_c2xc32_l2 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c2xc32, atoms,
                                                                      halfatoms)

    find_ind(atoms_c31_l1, atoms_l1, halfatoms, 0, 0, inds, 0)
    find_ind(atoms_c31_l2, atoms_l2, halfatoms, 1, 1, inds, 0)

    find_ind(atoms_c32_l1, atoms_l1, halfatoms, 0, 0, inds, 1)
    find_ind(atoms_c32_l2, atoms_l2, halfatoms, 1, 1, inds, 1)

    find_ind(atoms_c2x_l1, atoms_l2, halfatoms, 1, 0, inds, 2)
    find_ind(atoms_c2x_l2, atoms_l1, halfatoms, 0, 1, inds, 2)

    find_ind(atoms_c2xc31_l1, atoms_l2, halfatoms, 1, 0, inds, 3)
    find_ind(atoms_c2xc31_l2, atoms_l1, halfatoms, 0, 1, inds, 3)

    find_ind(atoms_c2xc32_l1, atoms_l2, halfatoms, 1, 0, inds, 4)
    find_ind(atoms_c2xc32_l2, atoms_l1, halfatoms, 0, 1, inds, 4)

    for i in range(natoms):
        dis1 = np.linalg.norm(atoms[i]-atoms_c31[inds[i][0]])
        dis2 = np.linalg.norm(atoms[i]-atoms_c32[inds[i][1]])
        dis3 = np.linalg.norm(atoms[i]-atoms_c2x[inds[i][2]])
        dis4 = np.linalg.norm(atoms[i]-atoms_c2xc31[inds[i][3]])
        dis5 = np.linalg.norm(atoms[i]-atoms_c2xc32[inds[i][4]])

        #print("check")
        if dis1>1E-10:
            print(i, atoms[i], atoms_c31[inds[i][0]], atoms_c32[inds[i][1]])
        if dis2>1E-10:
            print(i, atoms[i], atoms_c31[inds[i][0]], atoms_c32[inds[i][1]])
        if dis3>1E-10:
            print(i, atoms[i], atoms_c31[inds[i][0]], atoms_c2x[inds[i][2]])
        if dis4>1E-10:
            print("alert", i)
        if dis5>1E-10:
            print("alert", i)

    print(inds[0])
    print(inds[1])
    np.save("../data/group/new"+str(n_moire)+".npy", inds)
    # print("="*50)
