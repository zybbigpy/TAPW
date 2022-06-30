import numpy as np

import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
from mtbmtbg.moire_shuffle import cont_shuffle_to_tbplw
from mtbmtbg.config import Cont, Structure

# reciprocal unit vector for atom system
A_G_UNITVEC_1 = Structure.A_G_UNITVEC_1
A_G_UNITVEC_2 = Structure.A_G_UNITVEC_2


def _set_kpt(rotmat):

    kpt = (-A_G_UNITVEC_1+A_G_UNITVEC_2)/3
    # print("kpt:", kpt)
    # after rotation
    kpt1 = kpt@rotmat.T
    kpt2 = kpt@rotmat

    return {'kpt1': kpt1, 'kpt2': kpt2}


def _check_eq(vec1, vec2):

    assert vec1.shape == vec2.shape

    if np.linalg.norm(vec1-vec2)<1E-9:
        return True
    else:
        return False


def _set_g_vec_list_valley(n_moire: int, g_vec_list: np.ndarray, m_basis_vecs: dict, valley: int) -> np.ndarray:
    """set Glist containg one specific valley or all valleys

    Args:
        n_moire (int): an integer describe the moire system
        g_vec_list (np.ndarray): original Glist and G[0, 0] = [0, 0]
        m_basis_vecs (dict): moire basis vecs dictionary
        valley (ValleyType.VALLEYK1): valley

    Returns:
        np.ndarray: Glist for computation
    """

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    offset = n_moire*m_g_unitvec_1+n_moire*m_g_unitvec_2

    gv1 = g_vec_list+offset
    gv2 = g_vec_list-offset

    if valley == 1:
        return gv1
    elif valley == -1:
        return gv2
    else:  # default use VALLEYK1
        return gv1


def _make_transfer_const(m_basis_vecs, valley):

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']

    # three nearest g vec
    g1 = np.array([0, 0])
    g2 = -valley*m_g_unitvec_2
    g3 = valley*m_g_unitvec_1

    omega1, omega2 = np.exp(1j*2*np.pi/3)**valley, np.exp(-1j*2*np.pi/3)**valley

    t1 = np.array([[Cont.U1, Cont.U2], [Cont.U2, Cont.U1]])
    t2 = np.array([[Cont.U1, Cont.U2*omega2], [Cont.U2*omega1, Cont.U1]])
    t3 = t2.T

    return (g1, g2, g3, t1, t2, t3)


def _make_t(glist, m_basis_vecs, valley):
    """
    calculate interlayer interaction hamiltonian element
    """
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    glist_size = np.shape(glist)[0]

    tmat = np.zeros((2*glist_size, 2*glist_size), complex)
    (g1, g2, g3, t1, t2, t3) = _make_transfer_const(m_basis_vecs, valley)

    for i in range(glist_size):
        for j in range(glist_size):
            delta_k = glist[i]-glist[j]
            # matrix element in three cases:
            if _check_eq(delta_k, g1):
                tmat[2*i:2*i+2, 2*j:2*j+2] = t1
            if _check_eq(delta_k, g2):
                tmat[2*i:2*i+2, 2*j:2*j+2] = t2
            if _check_eq(delta_k, g3):
                tmat[2*i:2*i+2, 2*j:2*j+2] = t3

    return tmat


def _make_h(glist, k, kpt, rotmat, valley):
    """
    calculate first layer hamiltonian, approximated by dirac hamiltonian
    """

    glist_size = np.shape(glist)[0]
    h1mat = np.zeros((2*glist_size, 2*glist_size), complex)

    for i in range(glist_size):
        q = k+glist[i]-valley*kpt
        #q = q@rotmat
        dirac = Cont.HBARVF*(valley*Cont.SIGMA_X*q[1]-Cont.SIGMA_Y*q[0])
        h1mat[2*i:2*i+2, 2*i:2*i+2] = dirac

    return h1mat


def _make_hamk(k, kpts, glist, rt_mtrx_half, tmat, valley):
    """
    generate total hamiltonian 
    """
    kpt1 = kpts['kpt1']
    kpt2 = kpts['kpt2']

    h1mat = _make_h(glist, k, kpt1, rt_mtrx_half, valley)
    h2mat = _make_h(glist, k, kpt2, rt_mtrx_half.T, valley)
    hamk = np.block([[h1mat, np.conj(np.transpose(tmat))], [tmat, h2mat]])

    return hamk


def cont_solver(n_moire: int, n_g: int, n_k: int, disp: bool = True, valley: int = 1) -> dict:
    """
    continuum model solver for TBG system
    """

    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    # construct moire info
    rt_angle_r, rt_angle_d = mset._set_moire_angle(n_moire)
    rt_mtrx_half = mset._set_rt_mtrx(rt_angle_r/2)
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = _set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # interlayer interaction
    tmat = _make_t(g_vec_list, m_basis_vecs, valley)
    # atomic K points
    kpts = _set_kpt(rt_mtrx_half)

    if disp:
        (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
    else:
        kmesh = mgk.set_kmesh(n_k, m_basis_vecs)

    for k in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = _make_hamk(k, kpts, g_vec_list, rt_mtrx_half, tmat, valley)
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)

    return {'emesh': np.array(emesh), 'dmesh': np.array(dmesh), 'kline': kline}


def cont_potential(n_moire:int, n_g:int, valley: int = 1):


    # construct moire info
    rt_angle_r, rt_angle_d = mset._set_moire_angle(n_moire)
    rt_mtrx_half = mset._set_rt_mtrx(rt_angle_r/2)
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = _set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # interlayer interaction
    tmat = _make_t(g_vec_list, m_basis_vecs, valley)
    # atomic K points
    kpts = _set_kpt(rt_mtrx_half)


    hamk = _make_hamk(high_symm_pnts['gamma'], kpts, g_vec_list, rt_mtrx_half, tmat, valley)
    hamk_shuffled = cont_shuffle_to_tbplw(hamk)
    dim1 = int(hamk.shape[0]/2)
    dim2 = 2*dim1
    u = hamk_shuffled[0:dim1, dim1:dim2]
    dim1 = int(u.shape[0]/2)
    dim2 = 2*dim1
    u1 = np.abs(u[0:dim1, 0:dim1])[0, :]

    return {'glist': o_g_vec_list, 'mpot': u1}

