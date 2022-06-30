import time
import numpy as np
import scipy.linalg as sla
from scipy import sparse

import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
import mtbmtbg.moire_io as mio
from mtbmtbg.config import TBInfo, DataType, EngineType, ValleyType

VPI_0 = TBInfo.VPI_0
VSIGMA_0 = TBInfo.VSIGMA_0
R_RANGE = TBInfo.R_RANGE


def _set_g_vec_list_valley(n_moire: int, g_vec_list: np.ndarray, m_basis_vecs: dict,
                           valley: ValleyType.VALLEYK1) -> np.ndarray:
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
    gvc = np.append(gv1, gv2, axis=0)

    if valley == ValleyType.VALLEYK1:
        return gv1
    elif valley == ValleyType.VALLEYK2:
        return gv2
    elif valley == ValleyType.VALLEYC:
        return gvc
    else:  # default use VALLEYK1
        return gv1


def _sk_integral(ndist_dict: dict) -> np.ndarray:
    """calculate sk integral of neighour pair ri-rj

    Args:
        ndist_dict (dict): neighour distance dictionary

    Returns:
        np.ndarray: hopping array of (ri-rj)
    """
    dr = ndist_dict['dr']
    dd = ndist_dict['dd']

    res = np.sum(dr**2, axis=1)+dd**2
    res_sqrt = np.sqrt(res)

    vpi = VPI_0*np.exp(-(res_sqrt-mset.A_EDGE)/R_RANGE)
    vsigma = VSIGMA_0*np.exp(-(res_sqrt-mset.D_AB)/R_RANGE)

    hopping = vpi*(1-dd**2/res)+vsigma*(dd**2)/res

    return hopping


def _set_const_mtrx(
        n_moire: int,
        npair_dict: dict,
        ndist_dict: dict,
        m_basis_vecs: dict,
        g_vec_list: np.ndarray,
        atom_pstn_list: np.ndarray,
) -> dict:
    """setup constant matrix in calculating TBPLW

    Args:
        n_moire (int): an integer describing the moire system
        npair_dict (dict): neighbour pair dictionary
        ndist_dict (dict): neighbour distance dictionary
        m_basis_vecs (dict): moire basis vectors dictionary
        g_vec_list (np.ndarray): Glist (Attention! should be sampled near specific `VALLEY`)
        atom_pstn_list (np.ndarray): atom postions in a moire unit cell

    Raises:
        Exception: Hopping matrix is not Hermitian
        Exception: Overlap matrix is not Hermitian

    Returns:
       dict: {gr_mtrx, tr_mtrx, sr_mtrx}
    """

    # read values
    row, col = npair_dict['r'], npair_dict['c']
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    n_g = g_vec_list.shape[0]
    n_atom = atom_pstn_list.shape[0]
    # normalize factor
    factor = 1/np.sqrt(n_atom/4)

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2])) for g in g_vec_list for r in atom_pstn_list
                       ]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = sla.block_diag(g1, g2, g3, g4)

    hopping = _sk_integral(ndist_dict)
    tr_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))
    tr_mtrx_cc = (tr_mtrx.transpose()).conjugate()
    tr_mtrx_delta = tr_mtrx-tr_mtrx_cc

    if tr_mtrx_delta.max()>1.0e-9:
        print(tr_mtrx_delta.max())
        raise Exception("Hopping matrix is not hermitian?!")

    diag_ones = sparse.diags([1 for i in range(n_atom)])
    sr_mtrx = gr_mtrx@(diag_ones@(gr_mtrx.conj().T))
    sr_mtrx_cc = (sr_mtrx.transpose()).conjugate()
    sr_mtrx_delta = sr_mtrx-sr_mtrx_cc

    if sr_mtrx_delta.max()>1.0e-9:
        print(sr_mtrx_delta.max())
        raise Exception("Overlap matrix is not hermitian?!")

    const_mtrx_dict = {}
    const_mtrx_dict['gr'] = gr_mtrx
    const_mtrx_dict['tr'] = tr_mtrx
    const_mtrx_dict['sr'] = sr_mtrx

    return const_mtrx_dict


def _cal_eigen_hamk(hamk, smat, datatype=DataType.CORRU, engine=EngineType.TBPLW) -> tuple:
    """solve the eigenvalue problem using different engine according to engine and datatype

    Args:
        hamk (_type_): _description_
        smat (_type_): _description_
        datatype (_type_, optional): structure of input atoms. Defaults to DataType.CORRU.
        engine (_type_, optional): differnet TB engines. Defaults to EngineType.TBPLW.

    Returns:
        tuple: (v, w)
    """
    w = 0

    if engine == EngineType.TBFULL:
        v, _ = np.linalg.eigh(hamk)
    elif engine == EngineType.TBSPARSE:
        v, _ = sparse.linalg.eigs(hamk, k=10, sigma=0.78)
        v = np.real(v)
    else:  # default using TBPLW
        if datatype == DataType.RELAX:
            v, w = sla.eigh(hamk, b=smat)
        else:
            v, w = np.linalg.eigh(hamk)

    return (v, w)


def _cal_hamiltonian_k(ndist_dict: dict,
                       npair_dict: dict,
                       const_mtrx_dict: dict,
                       k_vec: np.ndarray,
                       n_atom: int,
                       engine=EngineType.TBPLW):
    """calculate hk

    Args:
        ndist_dict (dict): neighbour distance dictionary
        npair_dict (dict): neighbour pair dictionary   
        const_mtrx_dict (dict): const matrix dictionary 
        k_vec (np.ndarray): kpoint needed to be solved  
        n_atom (int): number of atoms in a moire unit cell  
        engine : Defaults to EngineType.TBPLW.

    Raises:
        Exception: kr matrix is not hermitian.
        Exception: FullTB matrix is not hermitian. 

    Returns:
        _type_: _description_
    """

    row, col = npair_dict['r'], npair_dict['c']
    gr_mtrx = const_mtrx_dict['gr']
    tr_mtrx = const_mtrx_dict['tr']
    dr = ndist_dict['dr']

    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))
    kr_mtrx_cc = (kr_mtrx.transpose()).conjugate()
    kr_mtrx_delta = kr_mtrx-kr_mtrx_cc

    if kr_mtrx_delta.max()>1.0e-9:
        print(kr_mtrx_delta.max())
        raise Exception("kr matrix is not hermitian?!")

    # Full tight binding spectrum can be calculated by directly diagonalized `hr_mtrx`
    hr_mtrx = kr_mtrx.multiply(tr_mtrx)
    hr_mtrx_cc = (hr_mtrx.transpose()).conjugate()
    hr_mtrx_delta = hr_mtrx-hr_mtrx_cc

    if hr_mtrx_delta.max()>1.0e-9:
        print(hr_mtrx_delta.max())
        raise Exception("fullTB matrix is not hermitian?!")

    if engine == EngineType.TBFULL:
        return hr_mtrx.toarray()
    elif engine == EngineType.TBSPARSE:
        return hr_mtrx
    else:  # default using TBPLW
        return gr_mtrx@(hr_mtrx@(gr_mtrx.conj().T))


def tb_solver(n_moire: int,
              n_g: int,
              n_k: int,
              disp: bool = True,
              datatype=DataType.CORRU,
              engine=EngineType.TBPLW,
              valley=ValleyType.VALLEYK1) -> dict:
    """tight binding solver for TBG

    Args:
        n_moire (int): an integer describing the size of commensurate TBG systems
        n_g (int): Glist size, n_g = 5 for MATBG
        n_k (int): n_k 
        disp (bool): whether calculate dispersion
        datatype (DataType, optional): atom data type. Defaults to DataType.CORRU.
        engine (EngineType, optional): TB solver engine type. Defaults to EngineType.TBPLW.
        valley (EngineType, optional): valley concerned. Defaults to EngineType.VALLEYK1.

    Returns:
        dict:         
        'emesh': np.array(emesh),
        'dmesh': np.array(dmesh),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    """
    start_time = time.process_time()
    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    # load atom data
    atom_pstn_list = mio.read_atom_pstn_list(n_moire, datatype)
    # construct moire info
    (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
    (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_basis_vecs)
    (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
    # set up g list
    o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
    # move to specific valley or combined valley
    g_vec_list = _set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
    # constant matrix dictionary
    const_mtrx_dict = _set_const_mtrx(n_moire, npair_dict, ndist_dict, m_basis_vecs, g_vec_list, atom_pstn_list)
    # constant list
    (transmat_list, neighbor_map) = mgk.set_kmesh_neighbour(n_g, m_basis_vecs, o_g_vec_list)

    if disp:
        (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
    else:
        kmesh = mgk.set_kmesh(n_k, m_basis_vecs)

    n_atom = atom_pstn_list.shape[0]
    n_band = g_vec_list.shape[0]*4
    n_kpts = kmesh.shape[0]
    print("="*100)
    print("num of atoms".ljust(30), ":", n_atom)
    print("num of kpoints".ljust(30), ":", n_kpts)
    print("num of bands".ljust(30), ":", n_band)
    print("="*100)
    setup_time = time.process_time()

    for k_vec in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = _cal_hamiltonian_k(ndist_dict, npair_dict, const_mtrx_dict, k_vec, n_atom, engine)
        eigen_val, eigen_vec = _cal_eigen_hamk(hamk, const_mtrx_dict['sr'], datatype, engine)
        if np.max(eigen_val)>emax:
            emax = np.max(eigen_val)
        if np.min(eigen_val)<emin:
            emin = np.min(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    comp_time = time.process_time()

    print("="*100)
    print("emax =", emax, "emin =", emin)
    print("="*100)
    print("set up time:", setup_time-start_time, "comp time:", comp_time-setup_time)
    print("="*100)

    return {
        'emesh': np.array(emesh),
        'dmesh': np.array(dmesh),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    }
