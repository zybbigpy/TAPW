import numpy as np

import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_tb as mtb
import mtbmtbg.moire_gk as mgk
import mtbmtbg.moire_io as mio
from mtbmtbg.config import TBInfo, DataType, EngineType, ValleyType


def _set_moire_potential(hamk: np.ndarray) -> np.ndarray:
    """get the moire potential part from hamk

    Args:
        hamk (np.ndarray): hk

    Returns:
        np.ndarray: 2d array of moire potential
    """
    dim1 = int(hamk.shape[0]/2)
    dim2 = 2*dim1
    # h1 = hamk[0:dim1, 0:dim1]
    # h2 = hamk[0:dim1, dim1:dim2]
    # h3 = hamk[dim1:dim2, 0:dim1]
    # h4 = hamk[dim1:dim2, dim1:dim2]
    h2 = hamk[0:dim1, dim1:dim2]
    h3 = hamk[dim1:dim2, 0:dim1]
    assert np.allclose(h2, h3.T.conj()) == True
    u = h2

    return u


def _analyze_moire_potential(u: np.ndarray) -> tuple:
    """get U_{A1, A2} U_{A1, B2} U_{B1, A2} U_{B1, B2} part of the moire potential        

    Args:
        u (np.ndarray): moire potential

    Returns:
        dict: {'u1': u1, 'u2': u2, 'u3': u3, 'u4': u4}
    """

    dim1 = int(u.shape[0]/2)
    dim2 = 2*dim1
    # U_{A1, A2} U_{A1, B2}
    u1 = u[0:dim1, 0:dim1]
    u2 = u[0:dim1, dim1:dim2]
    # U_{B1, A2} U_{B1, B2}
    u3 = u[dim1:dim2, 0:dim1]
    u4 = u[dim1:dim2, dim1:dim2]
    # reserve the coupling U_{G0, Gi}, make sure G0=[0,0]
    u1 = np.abs(u1[0, :])
    u2 = np.abs(u2[0, :])
    u3 = np.abs(u3[0, :])
    u4 = np.abs(u4[0, :])

    return {'u1': u1, 'u2': u2, 'u3': u3, 'u4': u4}


def analyze_moire_potential(n_moire: int, n_g: int, datatype=DataType.CORRU, valley=ValleyType.VALLEYK1) -> dict:
    """calculate the moire potential at high symmetry point

    Args:
        n_moire (int): an integer to describe the size of moire TBG .
        n_g (int): an interger to control the glist size. 
        datatype (DataType, optional): input atom type. Defaults to DataType.CORRU.
        valley (ValleyType, optional): valley to be calculated. Defaults to ValleyType.VALLEYK1.

    Returns:
        dict: { 'glist': o_g_vec_list, 'mpot': moire_potential }
    """

    # load atom data
    atom_pstn_list = mio.read_atom_pstn_list(n_moire, datatype)
    # construct moire info
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_basis_vecs)
    (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
    # set up original g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = mtb._set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # constant matrix dictionary
    const_mtrx_dict = mtb._set_const_mtrx(n_moire, npair_dict, ndist_dict, m_basis_vecs, g_vec_list, atom_pstn_list)
    # number of atoms in the moire unit cell
    n_atom = atom_pstn_list.shape[0]

    print("="*100)
    moire_potential = {}

    for kpnt in high_symm_pnts:
        print("analyze moire potential on high symmetry point:", kpnt, high_symm_pnts[kpnt])
        hamk = mtb._cal_hamiltonian_k(ndist_dict,
                                      npair_dict,
                                      const_mtrx_dict,
                                      high_symm_pnts[kpnt],
                                      n_atom,
                                      engine=EngineType.TBPLW)
        u = _set_moire_potential(hamk)
        moire_potential[kpnt] = _analyze_moire_potential(u)

    print("="*100)

    return {'glist': o_g_vec_list, 'mpot': moire_potential}


def analyze_band_convergence(n_moire: int, n_g: int, datatype=DataType.CORRU, valley=ValleyType.VALLEYK1) -> dict:
    """analyze band convergence by get the abs value of A1 bands.
    !ATTENTION!: ValleyType.VALLEYC is not supported HERE!

    Args:
        n_moire (int): an integer to describe the moire system.
        n_g (int): test the convergence of n_g
        datatype (DataType, optional): input atom data type. Defaults to DataType.CORRU.
        valley (ValleyType, optional): valley type. Defaults to ValleyType.VALLEYK1.
    
    Returns:
        dict:     return {'glist': o_g_vec_list, 'band': moire_band}
    """
    # load atom data
    atom_pstn_list = mio.read_atom_pstn_list(n_moire, datatype)
    # construct moire info
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_basis_vecs)
    (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
    # set up original g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = mtb._set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # constant matrix dictionary
    const_mtrx_dict = mtb._set_const_mtrx(n_moire, npair_dict, ndist_dict, m_basis_vecs, g_vec_list, atom_pstn_list)
    # number of atoms in the moire unit cell
    n_atom = atom_pstn_list.shape[0]

    print("="*100)
    moire_band = {}
    for kpnt in high_symm_pnts:
        print("analyze moire band convergence on high symmetry point:", kpnt, high_symm_pnts[kpnt])
        hamk = mtb._cal_hamiltonian_k(ndist_dict,
                                      npair_dict,
                                      const_mtrx_dict,
                                      high_symm_pnts[kpnt],
                                      n_atom,
                                      engine=EngineType.TBPLW)
        eigen_val, eigen_vec = mtb._cal_eigen_hamk(hamk, const_mtrx_dict['sr'], datatype, engine=EngineType.TBPLW)
        # choose one flat band.shape)
        n_band = eigen_vec.shape[0]
        flat_band = eigen_vec[:, n_band//2]
        # choose A1 component (wavefunction should be arranged in the order of A1[Gi], A2[Gi], B1[Gi], B2[Gi])
        flat_band_a1 = flat_band[:n_band//4]
        # np.abs is the same as calculating the |psi|^2
        moire_band[kpnt] = np.abs(flat_band_a1)

    return {'glist': o_g_vec_list, 'band': moire_band}
