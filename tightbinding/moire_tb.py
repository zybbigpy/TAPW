import numpy as np
import scipy as sp
import moire_setup as mset

from itertools import product

VPI_0 = 1
VSIGMA_0 = 1
R_RANGE = 1


def _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g):
    """
    set up g vector list
    """

    g_vec_list = []

    # construct a hexagon area by using three smallest g vectors 
    g_3 = -m_g_unitvec_1 - m_g_unitvec_2
    
    for (i, j) in product(range(n_g)):
        g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    for (i, j) in product(range(1, n_g)):
        g_vec_list.append(i*g_3 + j*m_g_unitvec_1)
    
    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(j*g_3 + i*m_g_unitvec_2)
     
    return g_vec_list


def _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k):
    
    k_step = 1/n_k
    
    kmesh = [i*k_step*m_g_unitvec_1 + j*k_step*m_g_unitvec_2 
             for (i, j) in product(range(n_k))]

    return kmesh


def _sk_integral(dr, dd):
    
    res = np.sum(dr**2, axis=1) + np.sum(dd*2, axis=1)
    res_sqrt = np.sqrt(res)

    vpi = VPI_0*np.exp(-(res_sqrt-mset.A_EDGE)/R_RANGE)
    vsigma = VSIGMA_0*np.exp(-(res_sqrt-mset.D_AB)/R_RANGE)

    hopping = vpi*(1-dd**2/res)+vsigma*(dd**2)/res

    return hopping


def _set_const_mtrx(n_moire, dr,  dd,  m_g_unitvec_1, m_g_unitvec_2, 
                    row,     col, g_vec_list, atom_pstn_list, valley):
    
    n_g = len(g_vec_list)
    n_atom = len(atom_pstn_list)

    gr_mtrx = np.zeros((n_g, n_atom), complex)
    factor = 1/np.sqrt(n_atom/4)
    offset = n_moire*m_g_unitvec_1+n_moire*m_g_unitvec_2
    
    for g in range(n_g):
        for atom in range(n_atom):
            r = atom_pstn_list(atom)
            gr_mtrx[g, atom] = factor*np.exp(-1j*np.dot(g_vec_list[g]+valley*offset, r))

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = sp.block_diag(g1, g2, g3, g4)    
    
    hopping = _sk_integral(dr, dd)
    tr_mtrx = sp.sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))

    return (gr_mtrx, tr_mtrx)


def cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom):
    
    tk_data = np.exp(-1j*np.dot(dr, k_vec))

    kr_mtrx = sp.sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))

    hr_mtrx = kr_mtrx.multiply(tr_mtrx)
    hamiltonian_k = gr_mtrx * (hr_mtrx * gr_mtrx.H)

    return hamiltonian_k


def tightbinding_solver(n_moire: int, n_g: int, n_k: int, valley: int):
    
    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
     m_k2_vec,      m_m_vec,     rt_mtrx_half) = mset._set_moire(n_moire)

    atom_pstn_list = mset.read_atom_pstn_list("../data/", n_moire)
    atom_neighbour_list = mset.read_atom_neighbour_list("../data/", n_moire)
    (dr, dd, row, col) = mset.set_relative_dis_ndarray(atom_pstn_list, atom_neighbour_list, m_g_unitvec_1, 
                                                       m_g_unitvec_2,  m_unitvec_1,         m_unitvec_2)

    n_atom = len(atom_pstn_list)
    kmesh = _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k)
    g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g)
    gr_mtrx, tr_mtrx = _set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
                                       row, col, g_vec_list, atom_pstn_list, valley)

    dmesh = []
    emesh = []

    for k_vec in kmesh:
        hamk = cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom)
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)

    return (emesh, kmesh)