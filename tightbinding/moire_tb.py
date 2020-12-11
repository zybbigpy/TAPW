import numpy as np
import tightbinding.moire_setup as mset

from scipy.linalg import block_diag
from scipy import sparse
from itertools import product

# eV
VPI_0 = -2.81
# eV
VSIGMA_0 = 0.48

R_RANGE = 0.184*mset.A_0


def _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g:int)->list:

    g_vec_list = []

    # construct a hexagon area by using three smallest g vectors 
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    for (i, j) in product(range(1, n_g), range(1, n_g)):
        g_vec_list.append(i*g_3 + j*m_g_unitvec_1)
    
    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(j*g_3 + i*m_g_unitvec_2)
     
    return g_vec_list


def _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k:int)->list:
    
    k_step = 1/n_k
    kmesh = [i*k_step*m_g_unitvec_1 + j*k_step*m_g_unitvec_2 
             for (i, j) in product(range(n_k), range(n_k))]

    return kmesh


def _sk_integral(dr, dd):
    """
    dr (*, 2) ndarray, dd (*, ) ndarray, * represents for interaction pair

    -------
    Return:
    
    hopping: (*, ) ndarray
    """
    res = np.sum(dr**2, axis=1) + dd**2
    res_sqrt = np.sqrt(res)

    vpi = VPI_0*np.exp(-(res_sqrt-mset.A_EDGE)/R_RANGE)
    vsigma = VSIGMA_0*np.exp(-(res_sqrt-mset.D_AB)/R_RANGE)

    hopping = vpi*(1-dd**2/res)+vsigma*(dd**2)/res

    return hopping


def _set_const_mtrx(n_moire,  dr,  dd,    m_g_unitvec_1,  m_g_unitvec_2, 
                    row, col, g_vec_list, atom_pstn_list, valley):
    """
    calculate two constant matrix

    --------
    Returns:
    
    1. gr_mtrx (4*n_g, n_atom) np.matrix

    2. tr_mtrx (n_atom, n_atom) sparse matrix
    """

    n_g = len(g_vec_list)
    n_atom = len(atom_pstn_list)
    
    factor = 1/np.sqrt(n_atom/4)
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g + valley*offset, r[:2]))
                        for g in g_vec_list for r in atom_pstn_list]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

    hopping = _sk_integral(dr, dd)
    tr_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))

    return (gr_mtrx, tr_mtrx)


def _cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom):
    
    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))
    hr_mtrx = kr_mtrx.multiply(tr_mtrx)

    hamiltonian_k = gr_mtrx * (hr_mtrx * gr_mtrx.H)

    return hamiltonian_k


def _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, nk):
    """
    moire dispertion, this code is just modifield from Prof Dai's realization
    """

    num_sec = 4
    ksec = np.zeros((num_sec,2),  float)
    num_kpt = nk * (num_sec - 1)
    kline = np.zeros((num_kpt),  float)
    kmesh = np.zeros((num_kpt,2),float)

    # set k path (K1 - Gamma - M - K2)
    ksec[0] = m_k1_vec
    ksec[1] = m_gamma_vec
    ksec[2] = m_m_vec
    ksec[3] = m_k2_vec

    for i in range(num_sec-1):
        vec = ksec[i+1] - ksec[i]
        klen = np.sqrt(np.dot(vec,vec))
        step = klen/(nk)

        for ikpt in range(nk):
            kline[ikpt+i*nk] = kline[i*nk-1] + ikpt * step   
            kmesh[ikpt+i*nk] = vec*ikpt/(nk-1) + ksec[i]

    return (kline, kmesh)


def tightbinding_solver(n_moire: int, n_g: int, n_k: int, valley: int, disp=False)->tuple:
    """  
    Tight binding solver for moire system

    -------
    Returns:
    
    1. emesh: eigenvalues, np.array(n_k, n_bands)
    2. dmesh: eigenvectors, np.array(n_k, n_bands, n_bands)
    3. kline: 0 when uniform sampling in 1st B.Z., k path for tb disp
    """
    
    
    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
     m_k2_vec,      m_m_vec,     rt_mtrx_half) = mset._set_moire(n_moire)

    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    atom_pstn_list = mset.read_atom_pstn_list("../data/", n_moire)
    atom_neighbour_list = mset.read_atom_neighbour_list("../data/", n_moire)
    (dr, dd, row, col) = mset.set_relative_dis_ndarray(atom_pstn_list, atom_neighbour_list, m_g_unitvec_1, 
                                                       m_g_unitvec_2,  m_unitvec_1,         m_unitvec_2)

    if(disp): # k-path sampling
        (kline, kmesh) = _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, n_k)
    else:     # uniform sampling
        kmesh = _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k)
    
    
    g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g)
    gr_mtrx, tr_mtrx = _set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
                                       row, col, g_vec_list, atom_pstn_list, valley)
    n_atom = len(atom_pstn_list)
    n_band = len(g_vec_list)*4
    n_kpts = len(kmesh)

    print('='*100)
    np.set_printoptions(6)
    print("atom unit vector".ljust(30), ":", mset.A_UNITVEC_1, mset.A_UNITVEC_2)
    print("atom reciprotocal unit vector".ljust(30), ":", mset.A_G_UNITVEC_1, mset.A_G_UNITVEC_2)
    print("moire unit vector".ljust(30), ":", m_unitvec_1, m_unitvec_2)
    print("moire recoprotocal unit vector".ljust(30), ":", m_g_unitvec_1, m_g_unitvec_2)
    print("num of atoms".ljust(30), ":", n_atom) 
    print("num of kpoints".ljust(30), ":", n_kpts)
    print("num of bands".ljust(30), ":", n_band)
    print('='*100)

    for k_vec in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = _cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom)
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        if np.max(eigen_val) > emax:
            emax = np.max(eigen_val)
        if np.min(eigen_val) < emin:
            emin = np.min(eigen_val)
        #print(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    print('='*100)
    print("emax =", emax, "emin =", emin)
    print('='*100)

    return (np.array(emesh), np.array(dmesh), kline)