import numpy as np
import tightbinding.moire_setup as mset
import matplotlib.pyplot as plt
import scipy.linalg as la

from scipy.linalg import block_diag
from scipy import sparse
from scipy.sparse.linalg import eigs
from itertools import product
from numba import jit


# eV
VPI_0 = -2.7
# eV
VSIGMA_0 = 0.48

R_RANGE = 0.184*mset.A_0


@jit
def _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int):
    """
    old version code, but it makes sure G[0]-offset=[0, 0].
    """
    g_vec_list = []

    # construct a hexagon area by using three smallest g vectors (with symmetry)
    # g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    # for (i, j) in product(range(n_g), range(n_g)):
    #     g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    # for (i, j) in product(range(1, n_g), range(1, n_g)):
    #     g_vec_list.append(i*g_3 + j*m_g_unitvec_1)
    
    # for i in range(n_g):
    #     for j in range(1, n_g):
    #         g_vec_list.append(j*g_3 + i*m_g_unitvec_2)

    for i in range(n_g):
        for j in range(n_g):
            g_vec_list.append(i*m_g_unitvec_1+j*m_g_unitvec_2)
    
    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(-j*m_g_unitvec_1+(i-j)*m_g_unitvec_2)

    for i in range(1, n_g):
        for j in range(1, n_g):
            g_vec_list.append(-i*m_g_unitvec_2+(j-i)*m_g_unitvec_1)
    
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2
     
    return np.array(g_vec_list)+offset*valley


def _set_g_vec_list_comb_valley(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int):
    """
    Glist constructed by combining two valleys
    """
    g_vec_list = []

    # construct a hexagon area by using three smallest g vectors (with symmetry)
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    for i in range(n_g):
        for j in range(n_g):
            g_vec_list.append(i*m_g_unitvec_1+j*m_g_unitvec_2)
    
    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(-j*m_g_unitvec_1+(i-j)*m_g_unitvec_2)

    for i in range(1, n_g):
        for j in range(1, n_g):
            g_vec_list.append(-i*m_g_unitvec_2+(j-i)*m_g_unitvec_1)
    
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2
    v1 = np.array(g_vec_list) + offset
    v2 = np.array(g_vec_list) - offset
     
    return np.append(v1, v2, axis=0)


def _set_g_vec_list_symm(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int):
    """
    new version code, but not in use, it cannot make sure G[0]-offset=[0, 0]
    """
    g_vec_list = []
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

    # construct a hexagon area by using three smallest g vectors (with symmetry)
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*g_3 + j*m_g_unitvec_1)
    
    for (i, j) in product(range(n_g), range(n_g)):
            g_vec_list.append(j*g_3 + i*m_g_unitvec_2)

    # remove repeated gvecs in glist
    g_vec_list = np.unique(np.array(g_vec_list), axis=0) + offset*valley

    return g_vec_list


@jit(nopython=True, parallel=True)
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
    3. sr_mtrx (4*n_g, 4*n_g)  np.matrix
    """

    n_g = len(g_vec_list)
    n_atom = len(atom_pstn_list)
    
    factor = 1/np.sqrt(n_atom/4)

    # in old version code, offset of Glist realized here
    # offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2
    # gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g + valley*offset, r[:2]))
    #                     for g in g_vec_list for r in atom_pstn_list]).reshape(n_g, n_atom)

    # new version code, in g list construction

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2]))
                        for g in g_vec_list for r in atom_pstn_list]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

    hopping = _sk_integral(dr, dd)
    tr_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))

    tr_mtrx_cc = (tr_mtrx.transpose()).conjugate()
    tr_mtrx_delta = tr_mtrx - tr_mtrx_cc
    
    if tr_mtrx_delta.max()>1.0E-9:
        print(tr_mtrx_delta.max())
        raise Exception("Tr matrix is not hermitian?!")    

    diag_ones = sparse.diags([1 for i in range(n_atom)])
    sr_mtrx = gr_mtrx * (diag_ones * gr_mtrx.H)

    sr_mtrx_cc = (sr_mtrx.transpose()).conjugate()
    sr_mtrx_delta = sr_mtrx - sr_mtrx_cc
    
    if sr_mtrx_delta.max()>1.0E-9:
        print(sr_mtrx_delta.max())
        raise Exception("Overlap matrix is not hermitian?!")  
    

    return (gr_mtrx, tr_mtrx, sr_mtrx)


def _cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom, fulltb):
    """
    Calculate H(k), we project the sparse hopping matrix on planewaves or fullTB
    """
    
    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))
    kr_mtrx_cc = (kr_mtrx.transpose()).conjugate()
    kr_mtrx_delta = kr_mtrx - kr_mtrx_cc

    if kr_mtrx_delta.max()>1.0E-9:
        print(kr_mtrx_delta.max())
        raise Exception("kr matrix is not hermitian?!")  

    # Full tight binding spectrum can be calculated by directly diagonalized `hr_mtrx``
    hr_mtrx = kr_mtrx.multiply(tr_mtrx)
    hr_mtrx_cc = (hr_mtrx.transpose()).conjugate()
    hr_mtrx_delta = hr_mtrx - hr_mtrx_cc
    
    if hr_mtrx_delta.max()>1.0E-9:
        print(hr_mtrx_delta.max())
        raise Exception("Hopping matrix is not hermitian?!")  

    if fulltb:
        # full TB
        hamk = hr_mtrx
    else:
        # planewave projection
        hamk = gr_mtrx * (hr_mtrx * gr_mtrx.H)

    print("check H(k) shape:", hamk.shape)
    return hamk
    

def _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k:int)->list:
    
    k_step = 1/n_k
    kmesh = [i*k_step*m_g_unitvec_1 + j*k_step*m_g_unitvec_2 
             for (i, j) in product(range(n_k), range(n_k))]

    return kmesh


def _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, nk):
    """
    moire dispertion, this code is just modifield from Prof Dai's realization.
    Note that, it is not a normal sampling on the kline.
    """

    num_sec = 4
    num_kpt = nk*(num_sec - 1)
    length  = 0
    
    klen  = np.zeros((num_sec),     float)
    ksec  = np.zeros((num_sec, 2),  float)
    kline = np.zeros((num_kpt+1),   float)
    kmesh = np.zeros((num_kpt+1, 2),float)

    # set k path (K1 - Gamma - M - K2)
    ksec[0] = m_k1_vec
    ksec[1] = m_gamma_vec
    ksec[2] = m_m_vec
    ksec[3] = m_k2_vec

    for i in range(num_sec-1):
        vec    = ksec[i+1] - ksec[i]
        length = np.sqrt(np.dot(vec,vec))
        klen[i+1] = klen[i] + length

        for ikpt in range(nk):
            kline[ikpt+i*nk] = klen[i] + ikpt*length/nk   
            kmesh[ikpt+i*nk] = ksec[i] + ikpt*vec/nk 
    kline[num_kpt] = kline[2*nk] + length
    kmesh[num_kpt] = ksec[3]

    return (ksec, kline, kmesh)


def _set_kmesh_neighbour(n_g, m_g_unitvec_1, m_g_unitvec_2):
    
    g_vec_list = []

    for i in range(n_g):
        for j in range(n_g):
            g_vec_list.append(i*m_g_unitvec_1+j*m_g_unitvec_2)
    
    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(-j*m_g_unitvec_1+(i-j)*m_g_unitvec_2)

    for i in range(1, n_g):
        for j in range(1, n_g):
            g_vec_list.append(-i*m_g_unitvec_2+(j-i)*m_g_unitvec_1)

    print("Gvec list[0] should be zero:", g_vec_list[0])
    num_g = len(g_vec_list)
    err = 0.02*np.dot(m_g_unitvec_1, m_g_unitvec_1)

    transmat_list = []
    for m in range(num_g):
        mat = np.zeros((num_g, num_g), float)
        q_vec = g_vec_list[m]
        for (i, j) in product(range(num_g), range(num_g)):
            diff_vec = g_vec_list[i] + q_vec - g_vec_list[j]
            diff = np.sqrt(np.dot(diff_vec, diff_vec))
            if diff<err:
                mat[j, i] = 1.0
        transmat = block_diag(mat, mat, mat, mat)
        transmat_list.append(transmat)

    neighbor_map = np.zeros((num_g, 2, 2), int)
    for m in range(num_g):
        q_vec = g_vec_list[m]
        for (i, j) in product(range(2), range(2)):
            neighbor_map[m, i, j] = -1
            neighbor_vec = i*m_g_unitvec_1+j*m_g_unitvec_2+q_vec
            for n in range(num_g):
                diff_vec = g_vec_list[n]-neighbor_vec
                diff = np.sqrt(np.dot(diff_vec, diff_vec))
                if diff<err:
                    neighbor_map[m, i, j] = n
    
    return transmat_list, neighbor_map


def _cal_eigen_hamk(hamk, smat, datatype, fulltb, sparse):
    """
    different method for eigenval problem
    """

    w = 0
    if fulltb:
        if sparse:
            print("Sparse TB Solver.")
            v, _ = eigs(hamk, k=10, sigma=0.78)
        else:
            print("Full TB. hamk shape", hamk.shape)
            v, _ = np.linalg.eigh(hamk.todense())
    else:
        if datatype == 'symm_relax' or datatype == 'relax':
            v, w = la.eigh(hamk, b=smat)
        else:
            v, w = np.linalg.eigh(hamk)

    return v, w


def tightbinding_solver(n_moire:int, n_g:int, n_k:int, datatype:str, valley:str, disp=True, fulltb=False, sparse=True)->tuple:
    """  
    Tight Binding Solver for moire system
    datatype support:'atomic', corrugation', 'relax', 'symm_relax'
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

    # load atom list
    atom_pstn_list = mset.read_atom_pstn_list(n_moire, datatype)
    # construct moire info
    (all_nns, enlarge_atom_pstn_list)= mset.set_atom_neighbour_list(atom_pstn_list, m_unitvec_1, m_unitvec_2)
    (dr, dd, row, col) = mset.set_relative_dis_ndarray_new(atom_pstn_list, enlarge_atom_pstn_list, all_nns)

    if(disp): 
        # k-path sampling
        (ksec, kline, kmesh) = _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, n_k)
    else:     
        # uniform sampling
        kmesh = _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k)
    
    if valley == 'valley_comb':
        g_vec_list = _set_g_vec_list_comb_valley(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire)
    elif valley == '+1':
        g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, 1)
    elif valley == '-1':
        g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, -1)
    else:
        g_vec_list = _set_g_vec_list_comb_valley(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire)
    
    # construct constant matrix
    (gr_mtrx, tr_mtrx, sr_mtrx)   = _set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
                                       row, col, g_vec_list, atom_pstn_list, valley)
    # construct constant list
    (transmat_list, neighbor_map) = _set_kmesh_neighbour(n_g, m_g_unitvec_1, m_g_unitvec_2)



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
        hamk = _cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom, fulltb)
        eigen_val, eigen_vec = _cal_eigen_hamk(hamk, sr_mtrx, datatype, fulltb, sparse)
        if np.max(eigen_val) > emax:
            emax = np.max(eigen_val)
        if np.min(eigen_val) < emin:
            emin = np.min(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    print('='*100)
    print("emax =", emax, "emin =", emin)
    print('='*100)

    return (np.array(emesh), np.array(dmesh), kline, transmat_list, neighbor_map)


def _set_moire_potential(hamk):
    dim1 = int(hamk.shape[0]/2)
    dim2 = 2*dim1
    h1 = hamk[0:dim1   , 0:dim1   ]
    h2 = hamk[0:dim1   , dim1:dim2]
    h3 = hamk[dim1:dim2, 0:dim1   ]
    h4 = hamk[dim1:dim2, dim1:dim2]
    u  = h2
    
    return u


def _analyze_moire_potential(u):
    dim1 = int(u.shape[0]/2)
    dim2 = 2*dim1
    print("dim1", dim1)
    u1 = u[0:dim1   , 0:dim1   ]
    u2 = u[0:dim1   , dim1:dim2]
    u3 = u[dim1:dim2, 0:dim1   ]
    u4 = u[dim1:dim2, dim1:dim2]
    moire_potential = u1+u2+u3+u4
    print(u1.shape)
    
    return u1


def moire_analyze(n_moire:int, n_g:int, valley:int, datatype:str)->tuple:
    
    
    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
     m_k2_vec,      m_m_vec,     rt_mtrx_half) = mset._set_moire(n_moire)


    atom_pstn_list = mset.read_atom_pstn_list(n_moire, datatype)


    (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_unitvec_1, m_unitvec_2)
    (dr, dd, row, col) = mset.set_relative_dis_ndarray_new(atom_pstn_list, enlarge_atom_pstn_list, all_nns)

 
    (ksec, kline, kmesh) = _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, 10)

    

    g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley)
    gr_mtrx, tr_mtrx, sr_mtrx = _set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
                                                row, col, g_vec_list, atom_pstn_list, valley)


    n_atom = len(atom_pstn_list)
    n_band = len(g_vec_list)*4
    potential = []

    print('='*100)
    np.set_printoptions(6)
    print("atom unit vector".ljust(30), ":", mset.A_UNITVEC_1, mset.A_UNITVEC_2)
    print("atom reciprotocal unit vector".ljust(30), ":", mset.A_G_UNITVEC_1, mset.A_G_UNITVEC_2)
    print("moire unit vector".ljust(30), ":", m_unitvec_1, m_unitvec_2)
    print("moire recoprotocal unit vector".ljust(30), ":", m_g_unitvec_1, m_g_unitvec_2)
    print("num of atoms".ljust(30), ":", n_atom) 
    print("num of bands".ljust(30), ":", n_band)
    print('='*100)

    for kpts in ksec:
        hamk = _cal_hamiltonian_k(dr, kpts, gr_mtrx, tr_mtrx, row, col, n_atom, False)
        u    = _set_moire_potential(hamk)
        print("max u", np.max(u))
        pot  = _analyze_moire_potential(u)
        potential.append(pot)

    offset = valley*(n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2)
    g_vec_list = g_vec_list - offset
    print("Gvec list[0] should be zero:", g_vec_list[0])

    return ksec, g_vec_list, np.array(potential)