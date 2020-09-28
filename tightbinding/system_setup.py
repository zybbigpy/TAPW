import numpy as np

from itertools import product

# lattice constant
a_0 = 5

# unit vector for atom system
a_unitvec_1 = np.array([1, 2])
a_unitvec_2 = np.array([2, 3])

# reciprocal unit vector for atom system
a_g_unitvec_1 = np.array([3, 4])
a_g_unitvec_2 = np.array([5, 6])

# atom postion in graphene
atom_pstn_1 = np.array([0, 0])
atom_pstn_2 = np.array([1, 2])


def set_moire_angle(n_moire: int)->float:
    """
    get the angle by defining the moire number n_moire

    -----------
    Parameters:

    n_moire: moire number N

    ----------
    Return:
    moire angle in radius
    """
    return np.arcsin(np.sqrt(3)*(2*n_moire+1)/(6*n_moire**2+6*n_moire+2))

def set_rt_mtrx(theta: float):
    """
    create the rotation matrix

    -----------
    Parameters:

    theta: rotation angle in radius

    --------
    Returns:

    rt_mtrx: numpy array with shape (2, 2)
    """

    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    return rt_mtrx

def set_moire(n_moire: int)->tuple:
    """
    set up the parameters for the moire system

    """
    rt_angle = set_moire_angle(n_moire)
    rt_mtrx = set_rt_mtrx(rt_angle)
    rt_mtrx_half = set_rt_mtrx(rt_angle/2)

    # first `m_` represents for moire
    # moire unit vector
    m_unitvec_1 = (-(2*n_moire+1)*a_unitvec_1 + (n_moire +1)*a_unitvec_2)@rt_mtrx.T
    m_unitvec_2 = (-n_moire*a_unitvec_2 + (2*n_moire +1)*a_unitvec_2)@rt_mtrx.T
    m_unitvec_list = [m_unitvec_1, m_unitvec_2]
    
    # moire reciprocal vector
    m_g_unitvec_1 = a_g_unitvec_1@rt_mtrx_half.T - a_g_unitvec_1@rt_mtrx_half
    m_g_unitvec_2 = a_g_unitvec_2@rt_mtrx_half.T - a_g_unitvec_2@rt_mtrx_half
    m_g_unitvec_list = [m_g_unitvec_1, m_g_unitvec_2] 
    
    # high symmetry points
    m_gamma_vec = np.array([0, 0])
    m_k1_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_2/3
    m_k2_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_1/3
    m_m_vec = (m_k1_vec + m_k2_vec)/2
    symm_kpts_list = [m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec]

    return (m_unitvec_list, m_g_unitvec_list, symm_kpts_list)

def set_atom_list(n_moire: int):
    pass

def set_atom_neighbour(distance: float):
    pass

