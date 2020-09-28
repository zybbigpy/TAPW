import numpy as np

# lattice constant
a_0 = 5

# unit vector for atom system
a_unitvec_1 = np.array([1, 2])
a_unitvec_2 = np.array([2, 3])
a_unitvec_list = [a_unitvec_1, a_unitvec_2]

# reciprocal unit vector for atom system
a_g_unitvec_1 = np.array([3, 4])
a_g_unitvec_2 = np.array([5, 6])
a_g_unitvec_list = [a_g_unitvec_1, a_g_unitvec_2]

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

def create_rt_mtrx(theta: float):
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

def setup_moire(n_moire: int)->tuple:
    pass

def create_atom_list():
    pass

def find_atom_neighbour():
    pass

