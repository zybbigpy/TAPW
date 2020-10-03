import numpy as np
import scipy as sp
import moire_setup as mset

from itertools import product

VPI_0 = 1
VSIGMA_0 = 1


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


def _sk_integral(dr ,dd):
    pass


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


def _set_krelated_mtrx():
    pass


def cal_hamiltonian_k():
    pass


def tightbinding_solver(n_moire: int):
    pass