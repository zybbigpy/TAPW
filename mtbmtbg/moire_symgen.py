import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_io as mio
from mtbmtbg.config import DataType

import numpy as np
from scipy.spatial import cKDTree

c31_z = mset._set_rt_mtrx(np.pi*2/3)
c32_z = mset._set_rt_mtrx(np.pi*4/3)


def symm_c31_z(atom_pstn_2d: np.ndarray) -> np.ndarray:

    return atom_pstn_2d@c31_z


def symm_c32_z(atom_pstn_2d: np.ndarray) -> np.ndarray:

    return atom_pstn_2d@c32_z


def symm_reconstruct(m_basis_vecs: dict, atoms_pstn_2d: np.ndarray, symm_opt) -> np.ndarray:

    delta = 0.0001
    n_atoms = atoms_pstn_2d.shape[0]
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    m_unitvec_1 = m_basis_vecs['mu1']
    m_unitvec_2 = m_basis_vecs['mu2']

    atoms_pstn_2d_symm = symm_opt(atoms_pstn_2d)

    atoms_pstn_2d_symm_x = np.floor(np.dot(atoms_pstn_2d_symm, m_g_unitvec_1)/(np.pi*2)+delta).reshape(n_atoms, 1)
    atoms_pstn_2d_symm_y = np.floor(np.dot(atoms_pstn_2d_symm, m_g_unitvec_2)/(np.pi*2)+delta).reshape(n_atoms, 1)

    atoms_symm_new = atoms_pstn_2d_symm-(atoms_pstn_2d_symm_x)*m_unitvec_1-(atoms_pstn_2d_symm_y)*m_unitvec_2

    return atoms_symm_new


def find_group_ind(atoms_pstn_2d: np.ndarray, atoms_pstn_2d_symm: np.ndarray) -> np.ndarray:

    assert atoms_pstn_2d.shape == atoms_pstn_2d_symm.shape
    n_atoms = atoms_pstn_2d.shape[0]

    atoms_pstn_2d_l1 = atoms_pstn_2d[:n_atoms//2, :]
    atoms_pstn_2d_l2 = atoms_pstn_2d[n_atoms//2:, :]
    atoms_pstn_2d_symm_l1 = atoms_pstn_2d_symm[:n_atoms//2, :]
    atoms_pstn_2d_symm_l2 = atoms_pstn_2d_symm[n_atoms//2:, :]

    tree_l1 = cKDTree(atoms_pstn_2d_symm_l1)
    tree_l2 = cKDTree(atoms_pstn_2d_symm_l2)

    nn_l1 = tree_l1.query(atoms_pstn_2d_l1)
    assert (np.max(nn_l1[0])<1e-9)
    nn_l2 = tree_l2.query(atoms_pstn_2d_l2)
    assert (np.max(nn_l2[0])<1e-9)
    nn = np.append(nn_l1[1], nn_l2[1]+n_atoms//2)

    return nn


def cal_c3_group(n_moire: int, save: bool = False, path: str = "./") -> tuple:
    """generate indices for corresponding atoms after C3 symmetry operatiion

    Args:
        n_moire (int): an integer to describe the moire system
        save (bool, optional): whether to save .npy file. Defaults to False.
        path (str, optional): location to save .npy file. Defaults to "./".
    Returns:
        tuple: (nn_c31, nn_c32)
    """

    atoms_pstn_list = mio.read_atom_pstn_list(n_moire, DataType.RIGID)
    n_atoms = atoms_pstn_list.shape[0]
    (_, m_basis_vecs, _) = mset._set_moire(n_moire)

    atoms_pstn_2d = atoms_pstn_list[:, :2]
    atoms_pstn_2d_c31 = symm_reconstruct(m_basis_vecs, atoms_pstn_2d, symm_c31_z)
    atoms_pstn_2d_c32 = symm_reconstruct(m_basis_vecs, atoms_pstn_2d, symm_c32_z)

    nn_c31 = find_group_ind(atoms_pstn_2d, atoms_pstn_2d_c31)
    nn_c32 = find_group_ind(atoms_pstn_2d, atoms_pstn_2d_c32)

    for i in range(n_atoms):
        assert (np.allclose(atoms_pstn_2d[i], atoms_pstn_2d_c31[nn_c31[i]]))
    for i in range(n_atoms):
        assert (np.allclose(atoms_pstn_2d[i], atoms_pstn_2d_c32[nn_c32[i]]))

    if save:
        np.save("moire"+str(n_moire)+"_groupc31.npy", nn_c31)
        np.save("moire"+str(n_moire)+"_groupc32.npy", nn_c32)

    return (nn_c31, nn_c32)
