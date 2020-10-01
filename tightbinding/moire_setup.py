import numpy as np

from itertools import product

# lattice constant
a_0 = 2.46
a_edge = a_0 / np.sqrt(3)

# unit vector for atom system
a_unitvec_1 = np.array([np.sqrt(3)*a_0/2, -a_0/2])
a_unitvec_2 = np.array([np.sqrt(3)*a_0/2,  a_0/2])

# reciprocal unit vector for atom system
a_g_unitvec_1 = np.array([2*np.pi/(3*a_edge), -2*np.pi/(np.sqrt(3)*a_edge)])
a_g_unitvec_2 = np.array([2*np.pi/(3*a_edge),  2*np.pi/(np.sqrt(3)*a_edge)])

# atom postion in graphene
atom_pstn_1 = np.array([0, 0])
atom_pstn_2 = np.array([2*a_0/np.sqrt(3), 0])


def _set_moire_angle(n_moire: int)->float:
    """
    get the angle by defining the moire number n_moire

    ----------
    Return:
    moire angle in radius
    """
    return np.arcsin(np.sqrt(3)*(2*n_moire+1)/(6*n_moire**2+6*n_moire+2))


def _set_rt_mtrx(theta: float):
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


def _set_moire(n_moire: int)->tuple:
    """
    set up the parameters for the moire system

    -------
    Returns:

    parameter tuple

    """

    rt_angle = _set_moire_angle(n_moire)
    rt_mtrx = _set_rt_mtrx(rt_angle)
    rt_mtrx_half = _set_rt_mtrx(rt_angle/2)

    # first `m_` represents for moire
    # moire unit vector
    m_unitvec_1 = (-n_moire*a_unitvec_1 + (2*n_moire +1)*a_unitvec_2)@rt_mtrx_half.T
    m_unitvec_2 = (-(2*n_moire+1)*a_unitvec_1 + (n_moire +1)*a_unitvec_2)@rt_mtrx_half.T
    
    # moire reciprocal vector
    m_g_unitvec_1 = a_g_unitvec_1@rt_mtrx_half.T - a_g_unitvec_1@rt_mtrx_half
    m_g_unitvec_2 = a_g_unitvec_2@rt_mtrx_half.T - a_g_unitvec_2@rt_mtrx_half
    
    # high symmetry points
    m_gamma_vec = np.array([0, 0])
    m_k1_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_2/3
    m_k2_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_1/3
    m_m_vec = (m_k1_vec + m_k2_vec)/2

    return (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
            m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
            m_k2_vec,      m_m_vec,     rt_mtrx_half)


def set_atom_list(n_moire: int):

    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
     m_k2_vec,      m_m_vec,     rt_mtrx_half) =_set_moire(n_moire)

    ly = m_unitvec_1[1]
    n  = int(2*ly/a_0)+2
    atom_b_pstn = atom_pstn_2 - a_unitvec_1
    delta = 0.0001

    atom_list = []

    num_a1 = num_b1 = num_a2 = num_b2 =0

    # A1 atoms
    for (ix, iy) in product(range(n), range(n)):
        R = -ix*a_unitvec_1 + iy*a_unitvec_2
        R = R @ rt_mtrx_half.T
        x = R.dot(m_g_unitvec_1)/(2*np.pi)
        y = R.dot(m_g_unitvec_2)/(2*np.pi)
        if (x>-delta) and (x<(1-delta)) and (y>-delta) and (y<(1-delta)):
            atom_list.append(R)
            num_a1 += 1

    
    # B1 atoms
    for (ix, iy) in product(range(n), range(n)):
        R = -ix*a_unitvec_1 + iy*a_unitvec_2 + atom_b_pstn
        R = R @ rt_mtrx_half.T
        x = R.dot(m_g_unitvec_1)/(2*np.pi)
        y = R.dot(m_g_unitvec_2)/(2*np.pi)
        if (x>-delta) and (x<(1-delta)) and (y>-delta) and (y<(1-delta)):
            atom_list.append(R)
            num_b1 += 1


    # A2 atoms
    for (ix, iy) in product(range(n), range(n)):
        R = -ix*a_unitvec_1 + iy*a_unitvec_2
        R = R @ rt_mtrx_half
        x = R.dot(m_g_unitvec_1)/(2*np.pi)
        y = R.dot(m_g_unitvec_2)/(2*np.pi)
        if (x>-delta) and (x<(1-delta)) and (y>-delta) and (y<(1-delta)):
            atom_list.append(R)
            num_a2 += 1
            

    # B2 atoms
    for (ix, iy) in product(range(n), range(n)):
        R = -ix*a_unitvec_1 + iy*a_unitvec_2 + atom_b_pstn
        R = R @ rt_mtrx_half
        x = R.dot(m_g_unitvec_1)/(2*np.pi)
        y = R.dot(m_g_unitvec_2)/(2*np.pi)
        if (x>-delta) and (x<(1-delta)) and (y>-delta) and (y<(1-delta)):
            atom_list.append(R)
            num_b2 += 1

    assert(num_a1 == num_a2 == num_b1 == num_b2)
    
    return atom_list

def set_atom_neighbour(distance: float):
    pass

def get_atom_pos_neighour_list():
    pass

def set_relative_dis_ndarray(atom_pstn_list: list, atom_neighbour_list: list):
    """
    construct relative distance ndarry

    ------
    Returns
    """

    pass

def system_info_log(n_moire: int):

    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,      
     m_k2_vec,      m_m_vec,     rt_mtrx_half) = _set_moire(n_moire)
    
    np.set_printoptions(6)
    print("atom unit vector".ljust(30), ":", a_unitvec_1, a_unitvec_2)
    print("atom reciprotocal unit vector".ljust(30), ":", a_g_unitvec_1, a_g_unitvec_2)
    print("moire unit vector".ljust(30), ":", m_unitvec_1, m_unitvec_2)
    print("moire recoprotocal unit vector".ljust(30), ":", m_g_unitvec_1, m_g_unitvec_2)