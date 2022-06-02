import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import matplotlib.pyplot as plt
import numpy as np


def set_rt_mtrx(theta: float):

    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return rt_mtrx


c31 = set_rt_mtrx(np.pi*2/3)
c32 = set_rt_mtrx(np.pi*4/3)
c6 = set_rt_mtrx(np.pi/3)
c2x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


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
    atoms_symm_2d = atoms_symm[:, :2]
    atoms_symm_z = (atoms_symm[:, 2]).reshape(2*halfatoms, 1)
    atoms_symm_x = np.floor(np.dot(atoms_symm_2d, m_g_unitvec_1)/(np.pi*2)+delta).reshape(natoms, 1)
    atoms_symm_y = np.floor(np.dot(atoms_symm_2d, m_g_unitvec_2)/(np.pi*2)+delta).reshape(natoms, 1)
    # move to the original unit cell
    atoms_symm_2d_new = atoms_symm_2d-(atoms_symm_x)*m_unitvec_1-(atoms_symm_y)*m_unitvec_2

    print(atoms_symm_2d_new.shape, atoms_symm_z.shape)
    return np.append(atoms_symm_2d_new, atoms_symm_z, axis=1)


moire_list = [i for i in range(30, 32, 2)]
for n_moire in moire_list:
    atoms = np.array(tbset.read_atom_pstn_list(n_moire, 'relax'))
    print(atoms.shape)
    natoms = atoms.shape[0]
    halfatoms = natoms//2
    print("num of atoms:", natoms)
    inds = np.load("../data/group/new"+str(n_moire)+".npy")
    print("index shape:", inds.shape)

    (m_unitvec_1, m_unitvec_2, m_g_unitvec_1, m_g_unitvec_2, m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec,
     rt_mtrx_half) = tbset._set_moire(n_moire)

    atoms_c31 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c31, atoms, halfatoms)
    atoms_c32 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c32, atoms, halfatoms)
    atoms_c2x = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c2x, atoms, halfatoms)
    #plot(atoms, atoms_c2x)
    atoms_c2xc31 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c2xc31, atoms, halfatoms)
    atoms_c2xc32 = symm_reconstruct(m_g_unitvec_1, m_g_unitvec_2, symm_c2xc32, atoms, halfatoms)

    atoms_symm = []
    for i in range(natoms):
        dis1 = np.linalg.norm(atoms[i]-atoms_c31[inds[i][0]])
        dis2 = np.linalg.norm(atoms[i]-atoms_c32[inds[i][1]])
        dis3 = np.linalg.norm(atoms[i]-atoms_c2x[inds[i][2]])
        dis4 = np.linalg.norm(atoms[i]-atoms_c2xc31[inds[i][3]])
        dis5 = np.linalg.norm(atoms[i]-atoms_c2xc32[inds[i][4]])

        #print("check")
        if dis1>1.5:
            print(i, atoms[i], atoms_c31[inds[i][0]], atoms_c32[inds[i][1]])
        if dis2>1.5:
            print(i, atoms[i], atoms_c31[inds[i][0]], atoms_c32[inds[i][1]])
        if dis3>1.5:
            print(i, atoms[i], atoms_c31[inds[i][0]], atoms_c2x[inds[i][2]])
        if dis4>1.5:
            print("alert", i)
        if dis5>1.5:
            print("alert", i)

        #print(atoms[i], atoms_c31[inds[i][0]], atoms_c32[inds[i][1]], atoms_c2x[inds[i][2]], atoms_c2xc31[inds[i][3]], atoms_c2xc32[inds[i][4]])

        new_coord = (atoms[i]+atoms_c31[inds[i][0]]+atoms_c32[inds[i][1]]+atoms_c2x[inds[i][2]]+
                     atoms_c2xc31[inds[i][3]]+atoms_c2xc32[inds[i][4]])/6
        atoms_symm.append(new_coord)

    np.savetxt("../data/relaxsymm/symmatom_new"+str(n_moire)+".csv",
               np.array(atoms_symm),
               header="Rx, Ry, d",
               delimiter=',')
    # print("="*50)
