import sys
sys.path.append("..")

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import matplotlib.pyplot as plt

import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
from mtbmtbg.config import EngineType, ValleyType, Phonon


def _set_g_vec_list_valley(n_moire: int, g_vec_list: np.ndarray, m_basis_vecs: dict,
                           valley: ValleyType.VALLEYK1) -> np.ndarray:
    """setup glist near the K valley or near Gamma point

    Args:
        n_moire (int): an integer to describe the moire system  
        g_vec_list (np.ndarray): original Glist and G[0, 0] = [0, 0]
        m_basis_vecs (dict): moire basis vectors    
        valley (ValleyType.VALLEYK1): K valley or Gamma

    Returns:
        np.ndarray: _description_
    """

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    offset = n_moire*m_g_unitvec_1+n_moire*m_g_unitvec_2

    gv1 = g_vec_list+offset
    gv2 = g_vec_list-offset

    if valley == ValleyType.VALLEYK1:
        return gv1
    elif valley == ValleyType.VALLEYK2:
        return gv2
    elif valley == ValleyType.VALLEYG:
        return g_vec_list
    else:  # default use VALLEYK1
        return gv1


def _set_relative_dis(atom_pstn_list: np.ndarray, m_basis_vecs: dict, npair_dict: dict) -> dict:
    """set relative distance for the neighbour pairs

    Args:
        atom_pstn_list (np.ndarray): atom postions in the moire primitive unit cell
        m_basis_vecs (dict): dictionary storing moire basis vectors 
        npair_dict (dict): neighbour pair dictionary (row, col)

    Returns:
        dict: (dr, dd)
    """

    row, col = npair_dict['r'], npair_dict['c']
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    m_unitvec_1 = m_basis_vecs['mu1']
    m_unitvec_2 = m_basis_vecs['mu2']

    atom_pstn_2darray = atom_pstn_list[row]
    atom_neighbour_2darray = atom_pstn_list[col]

    dr = (atom_pstn_2darray-atom_neighbour_2darray)[:, :2]
    dd = (atom_pstn_2darray-atom_neighbour_2darray)[:, -1]
    x = np.dot(dr, m_g_unitvec_1)/(2*np.pi)
    y = np.dot(dr, m_g_unitvec_2)/(2*np.pi)

    # reconstruct dr (tricky here)
    x = x-np.trunc(2*x)
    y = y-np.trunc(2*y)

    dr = (x.reshape(-1, 1))*m_unitvec_1+(y.reshape(-1, 1))*m_unitvec_2

    return {'dr': dr, 'dd': dd}


def _set_gr_mtrx(
        n_moire: int,
        npair_dict: dict,
        g_vec_list: np.ndarray,
        atom_pstn_list: np.ndarray,
) -> dict:
    """setup projection matrix in calculating TBPLW

    Args:
        n_moire (int): an integer describing the moire system
        npair_dict (dict): neighbour pair dictionary
        g_vec_list (np.ndarray): Glist (Attention! should be sampled near specific `VALLEY` or Gamma)
        atom_pstn_list (np.ndarray): atom postions in a moire primitive unit cell

    Returns:
       dict: {gr_mtrx, tr_mtrx, sr_mtrx}
    """

    # read values
    row, col = npair_dict['r'], npair_dict['c']
    n_g = g_vec_list.shape[0]
    n_atom = atom_pstn_list.shape[0]
    # normalize factor
    factor = 1/np.sqrt(n_atom/4)

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2])) for g in g_vec_list for r in atom_pstn_list
                       ]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = sla.block_diag(g1, g2, g3, g4, g1, g2, g3, g4, g1, g2, g3, g4)

    return gr_mtrx


def _read_fc_rc():
    row = np.load('row.npy')
    col = np.load('col.npy')
    fc = sp.load_npz('hopping.npz')
    fc_t = fc.transpose()
    fc_delta = fc-fc_t
    if fc_delta.max()>1.0e-9:
        print("force constant is not symmetric?!", fc_delta.max())

    return fc, {'r': row, 'c': col}


def _cal_dynamic_k(k_vec: np.ndarray,
                   ndist_dict: dict,
                   npair_dict: dict,
                   n_atom: int,
                   fc,
                   gr_mtrx,
                   engine=EngineType.TBPLW):

    row, col = npair_dict['r'], npair_dict['c']
    dr = ndist_dict['dr']

    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sp.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))
    kr_mtrx = sp.bmat([[kr_mtrx, kr_mtrx, kr_mtrx], [kr_mtrx, kr_mtrx, kr_mtrx], [kr_mtrx, kr_mtrx, kr_mtrx]]).tocsr()
    kr_mtrx_cc = (kr_mtrx.transpose()).conjugate()
    kr_mtrx_delta = kr_mtrx-kr_mtrx_cc

    if kr_mtrx_delta.max()>1.0e-9:
        print(kr_mtrx_delta.max())
        raise Exception("kr matrix is not hermitian?!")

    dynamic_k = kr_mtrx.multiply(fc)/Phonon.CARBON_MASS
    dynamic_k_cc = (dynamic_k.transpose()).conjugate()
    dynamic_k_delta = dynamic_k-dynamic_k_cc
    if dynamic_k_delta.max()>1.0e-9:
        print(dynamic_k_delta.max())
        raise Exception("dynamic matrix is not hermitian?!")

    if engine == EngineType.TBFULL:
        return dynamic_k.todense()
    elif engine == EngineType.TBPLW:
        return gr_mtrx@(dynamic_k@(gr_mtrx.conj().T))
    else:
        return dynamic_k.todense()


def _cal_eigen_dynamick(dynamic_k, engine=EngineType.TBPLW):

    w = 0
    if engine == EngineType.TBFULL:
        v, _ = np.linalg.eigh(dynamic_k)
    elif engine == EngineType.TBPLW:
        v, w = np.linalg.eigh(dynamic_k)

    return v, w


def phonon_solver(n_moire: int, n_g: int, n_k: int, engine=EngineType.TBPLW, valley=ValleyType.VALLEYK1):

    atom_pstn_list = np.loadtxt('rigid_atom6_origin.csv')

    ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = _set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    fc, npair_dict = _read_fc_rc()
    ndist_dict = _set_relative_dis(atom_pstn_list, m_basis_vecs, npair_dict)
    gr_mtrx = _set_gr_mtrx(n_moire, npair_dict, g_vec_list, atom_pstn_list)
    kline, kmesh = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)

    n_atom = atom_pstn_list.shape[0]
    emesh = []

    for k_vec in kmesh:
        print(k_vec.shape)
        dynamick = _cal_dynamic_k(k_vec, ndist_dict, npair_dict, n_atom, fc, gr_mtrx, engine)
        eig_val, eig_vec = np.linalg.eigh(dynamick)
        print(eig_val)
        emesh.append(np.sqrt(eig_val)*Phonon.VaspToTHz)

    return kline, np.array(emesh)


n_k = 30
kline, emesh = phonon_solver(6, 2, n_k, engine=EngineType.TBPLW, valley=ValleyType.VALLEYG)
bands = 30
fig, ax = plt.subplots()
ax.set_xticks([kline[0], kline[n_k], kline[2*n_k], kline[3*n_k]])
ax.set_xticklabels([r"$\bar{K}$", r"$\bar{\Gamma}$", r"$\bar{M}$", r"$\bar{K}^\prime$"])
ax.set_xlim(0, kline[-1])
ax.set_ylabel("Frequency (THz)")
ax.axvline(x=kline[0], color="black")
ax.axvline(x=kline[n_k], color="black")
ax.axvline(x=kline[2*n_k], color="black")
ax.axvline(x=kline[3*n_k], color="black")
for i in range(30):
    ax.plot(kline, emesh[:, i])
plt.show()
