import numpy as np


class Structure:
    """ parameters for graphene structure
    """
    # angstrom
    A_C = 2.4683456
    A_EDGE = A_C/np.sqrt(3)

    # moire information (angstrom)
    D1_LAYER = 3.433333333
    D2_LAYER = 0.027777778
    D_AB = 3.35

    # unit vector for atom system
    A_UNITVEC_1 = np.array([np.sqrt(3)*A_C/2, -A_C/2])
    A_UNITVEC_2 = np.array([np.sqrt(3)*A_C/2, A_C/2])

    # reciprocal unit vector for atom system
    A_G_UNITVEC_1 = np.array([2*np.pi/(3*A_EDGE), -2*np.pi/(np.sqrt(3)*A_EDGE)])
    A_G_UNITVEC_2 = np.array([2*np.pi/(3*A_EDGE), 2*np.pi/(np.sqrt(3)*A_EDGE)])

    # atom postion in graphene
    ATOM_PSTN_1 = np.array([0, 0])
    ATOM_PSTN_2 = np.array([2*A_C/np.sqrt(3), 0])


class TBInfo:
    """ parameters for SK tight binding scheme
    """
    # eV
    VPI_0 = -2.7
    # eV
    VSIGMA_0 = 0.48
    # Ang
    R_RANGE = 0.184*Structure.A_C


class DataType:
    """Different atomic data type
    """
    RIGID = 'rigid'
    RELAX = 'relax'
    CORRU = 'corrugation'


class EngineType:
    """name for different engine types
    """
    TBPLW = 'TB'
    TBFULL = 'tbfull'
    TBSPARSE = 'tbsparse'


class ValleyType:
    """different type of valleys
    """
    VALLEY1 = 'valley1'
    VALLEY2 = 'valley2'
    # combined two valleys
    VALLEYC = 'valleyc'


class Cont:
    # two paramters, unit eV (chiral limit U1 = U2)
    U1 = 0.0797
    U2 = 0.0975
    # pauli matrices
    SIGMA_X = np.array([[0, 1], [1, 0]])
    SIGMA_Y = np.array([[0, -1j], [1j, 0]])
    SIGMA_Z = np.array([[1, 0], [0, -1]])
    # fermi velocity 2.1354eV*a
    HBARVF = 2.1354*Structure.A_C
