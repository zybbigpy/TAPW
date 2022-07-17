import numpy as np

from itertools import product
from scipy.linalg import block_diag


def set_g_vec_list(n_g: int, m_basis_vecs: dict) -> np.ndarray:
    """generate G list

    Args:
        n_g (int): an integer to descirbe the glist area size
        m_basis_vecs (dict): moire basis vectors dictionary

    Returns:
        np.ndarray: glist
    """

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    g_vec_list = []

    #construct a hexagon area by using three smallest g vectors (with symmetry)
    # g_3 = -m_g_unitvec_1-m_g_unitvec_2

    # for (i, j) in product(range(n_g), range(n_g)):
    #     g_vec_list.append(i*m_g_unitvec_1+j*m_g_unitvec_2)

    # for (i, j) in product(range(1, n_g), range(1, n_g)):
    #     g_vec_list.append(i*g_3+j*m_g_unitvec_1)

    # for i in range(n_g):
    #     for j in range(1, n_g):
    #         g_vec_list.append(j*g_3+i*m_g_unitvec_2)

    for i in range(n_g):
        for j in range(n_g):
            g_vec_list.append(i*m_g_unitvec_1+j*m_g_unitvec_2)

    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(-j*m_g_unitvec_1+(i-j)*m_g_unitvec_2)

    for i in range(1, n_g):
        for j in range(1, n_g):
            g_vec_list.append(-i*m_g_unitvec_2+(j-i)*m_g_unitvec_1)

    return np.array(g_vec_list)


def set_kmesh(n_k: int, m_basis_vecs: dict) -> np.ndarray:
    """set up normal k points sampling in 1st B.Z

    Args:
        n_k (int): number of kpts should be n_k**2. 
        m_basis_vecs (dict): moire basis vectors dictionary

    Returns:
        np.ndarray: kmesh
    """

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    k_step = 1/n_k
    kmesh = [i*k_step*m_g_unitvec_1+j*k_step*m_g_unitvec_2 for (i, j) in product(range(n_k), range(n_k))]

    return np.array(kmesh)


def set_kmesh_dos(n_k: int, m_basis_vecs: dict) -> np.ndarray:
    """set up normal k points sampling in 1st B.Z for DOS Calculation

    Args:
        n_k (int): number of kpts should be n_k**2. 
        m_basis_vecs (dict): moire basis vectors dictionary

    Returns:
        np.ndarray: kmesh
    """

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    # Different from `set_kmesh' here
    k_step = 1/(n_k-1)
    kmesh = [i*k_step*m_g_unitvec_1+j*k_step*m_g_unitvec_2 for (i, j) in product(range(n_k), range(n_k))]

    return np.array(kmesh)


def set_tb_disp_kmesh(n_k: int, high_symm_pnts: dict) -> tuple:
    """setup kpath along high symmetry points in moire B.Z.

    Note that, the length of each path is not equal, so it is not a 
    normal sampling. The Kpath is K1 - Gamma - M - K2.

    Args:
        n_k (int): number of kpts on one path
        high_symm_pnts (dict): hcoordinates of high symmetry points

    Returns:
        tuple: (kline, kmesh)
    """

    num_sec = 4
    num_kpt = n_k*(num_sec-1)
    length = 0

    klen = np.zeros((num_sec), float)
    ksec = np.zeros((num_sec, 2), float)
    kline = np.zeros((num_kpt+1), float)
    kmesh = np.zeros((num_kpt+1, 2), float)

    # set k path (K1 - Gamma - M - K2)
    ksec[0] = high_symm_pnts['k1']
    ksec[1] = high_symm_pnts['gamma']
    ksec[2] = high_symm_pnts['m']
    ksec[3] = high_symm_pnts['k2']

    for i in range(num_sec-1):
        vec = ksec[i+1]-ksec[i]
        length = np.sqrt(np.dot(vec, vec))
        klen[i+1] = klen[i]+length

        for ikpt in range(n_k):
            kline[ikpt+i*n_k] = klen[i]+ikpt*length/n_k
            kmesh[ikpt+i*n_k] = ksec[i]+ikpt*vec/n_k
    kline[num_kpt] = kline[2*n_k]+length
    kmesh[num_kpt] = ksec[3]

    return (kline, kmesh)


def set_kmesh_neighbour(n_g: int, m_basis_vecs: dict, g_vec_list: np.ndarray) -> tuple:
    """hard to describe.....

    Args:
        n_g (int): an integer describing the size of G list
        m_basis_vecs (dict): moire basis vectors dictionary
        g_vec_list (np.ndarray): Glist

    Returns:
        tuple: (transmat_list, neighbor_map)
    """
    assert np.allclose(g_vec_list[0], np.array([0.0, 0.0]))

    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']

    num_g = g_vec_list.shape[0]
    # searching tolerance
    err = 0.02*np.dot(m_g_unitvec_1, m_g_unitvec_1)

    transmat_list = []
    for m in range(num_g):
        mat = np.zeros((num_g, num_g), float)
        q_vec = g_vec_list[m]
        for (i, j) in product(range(num_g), range(num_g)):
            diff_vec = g_vec_list[i]+q_vec-g_vec_list[j]
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

    return (np.array(transmat_list), neighbor_map)
