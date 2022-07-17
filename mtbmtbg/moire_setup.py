import numpy as np

from itertools import product
from sklearn.neighbors import KDTree
from mtbmtbg.config import Structure

# lattice constant (angstrom)
A_C = Structure.A_C
A_EDGE = Structure.A_EDGE

# moire information (angstrom)
D1_LAYER = Structure.D1_LAYER
D2_LAYER = Structure.D2_LAYER
D_AB = Structure.D_AB

# unit vector for atom system
A_UNITVEC_1 = Structure.A_UNITVEC_1
A_UNITVEC_2 = Structure.A_UNITVEC_2

# reciprocal unit vector for atom system
A_G_UNITVEC_1 = Structure.A_G_UNITVEC_1
A_G_UNITVEC_2 = Structure.A_G_UNITVEC_2

# atom postion in graphene
ATOM_PSTN_1 = Structure.ATOM_PSTN_1
ATOM_PSTN_2 = Structure.ATOM_PSTN_2


def _set_moire_angle(n_moire: int) -> tuple:
    """set moire angle in radius and degrees

    Args:
        n_moire (int): an integer to describe the moire system.

    Returns:
        tuple: (angle_r, angle_d)
    """

    angle_r = np.arcsin(np.sqrt(3)*(2*n_moire+1)/(6*n_moire**2+6*n_moire+2))
    angle_d = angle_r/np.pi*180

    return (angle_r, angle_d)


def _set_rt_mtrx(theta: float) -> np.ndarray:
    """set 2D rotation matrix 

    Args:
        theta (float): rotation angle in radius

    Returns:
        np.ndarray: 2x2 rotation matrix
    """
    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return rt_mtrx


def _set_moire(n_moire: int) -> tuple:
    """calculate moire information for the system

    Args:
        n_moire (int): an integer to describe moire tbg system

    Returns:
        tuple: a tuple of system information. 
        (rotation angle tuple, moire basis vectors, high symmetry points)
    """

    rt_angle_r, rt_angle_d = _set_moire_angle(n_moire)
    rt_mtrx = _set_rt_mtrx(rt_angle_r)
    rt_mtrx_half = _set_rt_mtrx(rt_angle_r/2)

    # first `m_` represents for moire
    # moire unit vector
    m_unitvec_1 = (-n_moire*A_UNITVEC_1+(2*n_moire+1)*A_UNITVEC_2)@rt_mtrx_half.T
    m_unitvec_2 = (-(2*n_moire+1)*A_UNITVEC_1+(n_moire+1)*A_UNITVEC_2)@rt_mtrx_half.T

    # moire reciprocal vector
    m_g_unitvec_1 = A_G_UNITVEC_1@rt_mtrx_half.T-A_G_UNITVEC_1@rt_mtrx_half
    m_g_unitvec_2 = A_G_UNITVEC_2@rt_mtrx_half.T-A_G_UNITVEC_2@rt_mtrx_half

    # high symmetry points
    m_gamma_vec = np.array([0, 0])
    m_k1_vec = (m_g_unitvec_1+m_g_unitvec_2)/3+m_g_unitvec_2/3
    m_k2_vec = (m_g_unitvec_1+m_g_unitvec_2)/3+m_g_unitvec_1/3
    m_m_vec = (m_k1_vec+m_k2_vec)/2

    m_basis_vecs = {}
    m_basis_vecs['mu1'] = m_unitvec_1
    m_basis_vecs['mu2'] = m_unitvec_2
    m_basis_vecs['mg1'] = m_g_unitvec_1
    m_basis_vecs['mg2'] = m_g_unitvec_2
    high_symm_pnts = {}
    high_symm_pnts['gamma'] = m_gamma_vec
    high_symm_pnts['k1'] = m_k1_vec
    high_symm_pnts['k2'] = m_k2_vec
    high_symm_pnts['m'] = m_m_vec

    return ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts)


def set_atom_pstn_list(n_moire: int, corru: bool=True) -> np.ndarray:
    """generate all atom positions in a commesurate moire systems

    Args:
        n_moire (int): an integer to describe a commesurate moire tbg structure.
        corru (bool): genarate corrugation data or not, Default: True. 

    Returns:
        np.ndarray: atom positions
    """

    (rt_angle_r, _), m_basis_vecs, _ = _set_moire(n_moire)
    m_g_unitvec_1 = m_basis_vecs['mg1']
    m_g_unitvec_2 = m_basis_vecs['mg2']
    m_unitvec_1 = m_basis_vecs['mu1']
    rt_mtrx_half = _set_rt_mtrx(rt_angle_r/2)

    atom_b_pstn = ATOM_PSTN_2-A_UNITVEC_1
    small_g_vec = np.array([m_g_unitvec_1, m_g_unitvec_2, -m_g_unitvec_1-m_g_unitvec_2])

    # searching boundary
    ly = m_unitvec_1[1]
    n = int(2*ly/A_C)+2
    delta = 0.0001

    atom_pstn_list = []
    num_a1 = num_b1 = num_a2 = num_b2 = 0

    # find A1 atoms
    for (ix, iy) in product(range(n), range(n)):
        atom_pstn = -ix*A_UNITVEC_1+iy*A_UNITVEC_2
        atom_pstn = atom_pstn@rt_mtrx_half.T
        x = atom_pstn.dot(m_g_unitvec_1)/(2*np.pi)
        y = atom_pstn.dot(m_g_unitvec_2)/(2*np.pi)
        if (x> -delta) and (x<(1-delta)) and (y> -delta) and (y<(1-delta)):
            out_plane = D2_LAYER*np.sum(np.cos(np.dot(small_g_vec, atom_pstn)))
            d = 0.5*D1_LAYER+out_plane if corru else 0.5*D1_LAYER
            atom = np.array([atom_pstn[0], atom_pstn[1], d])
            atom_pstn_list.append(atom)
            num_a1 += 1

    # find B1 atoms
    for (ix, iy) in product(range(n), range(n)):
        atom_pstn = -ix*A_UNITVEC_1+iy*A_UNITVEC_2+atom_b_pstn
        atom_pstn = atom_pstn@rt_mtrx_half.T
        x = atom_pstn.dot(m_g_unitvec_1)/(2*np.pi)
        y = atom_pstn.dot(m_g_unitvec_2)/(2*np.pi)
        if (x> -delta) and (x<(1-delta)) and (y> -delta) and (y<(1-delta)):
            out_plane = D2_LAYER*np.sum(np.cos(np.dot(small_g_vec, atom_pstn)))
            d = 0.5*D1_LAYER+out_plane if corru else 0.5*D1_LAYER
            atom = np.array([atom_pstn[0], atom_pstn[1], d])
            atom_pstn_list.append(atom)
            num_b1 += 1

    # find A2 atoms
    for (ix, iy) in product(range(n), range(n)):
        atom_pstn = -ix*A_UNITVEC_1+iy*A_UNITVEC_2
        atom_pstn = atom_pstn@rt_mtrx_half
        x = atom_pstn.dot(m_g_unitvec_1)/(2*np.pi)
        y = atom_pstn.dot(m_g_unitvec_2)/(2*np.pi)
        if (x> -delta) and (x<(1-delta)) and (y> -delta) and (y<(1-delta)):
            out_plane = D2_LAYER*np.sum(np.cos(np.dot(small_g_vec, atom_pstn)))
            d = -0.5*D1_LAYER-out_plane if corru else -0.5*D1_LAYER
            atom = np.array([atom_pstn[0], atom_pstn[1], d])
            atom_pstn_list.append(atom)
            num_a2 += 1

    # find B2 atoms
    for (ix, iy) in product(range(n), range(n)):
        atom_pstn = -ix*A_UNITVEC_1+iy*A_UNITVEC_2+atom_b_pstn
        atom_pstn = atom_pstn@rt_mtrx_half
        x = atom_pstn.dot(m_g_unitvec_1)/(2*np.pi)
        y = atom_pstn.dot(m_g_unitvec_2)/(2*np.pi)
        if (x> -delta) and (x<(1-delta)) and (y> -delta) and (y<(1-delta)):
            out_plane = D2_LAYER*np.sum(np.cos(np.dot(small_g_vec, atom_pstn)))
            d = -0.5*D1_LAYER-out_plane if corru else -0.5*D1_LAYER
            atom = np.array([atom_pstn[0], atom_pstn[1], d])
            atom_pstn_list.append(atom)
            num_b2 += 1

    assert num_a1 == num_a2 == num_b1 == num_b2

    return np.array(atom_pstn_list)


def set_atom_neighbour_list(
        atom_pstn_list: np.ndarray,
        m_basis_vecs: dict,
        distance: float = 2.5113*A_C,
) -> tuple:
    """set atom neighbor information

    We adopt a KDTree searching scheme in a 3x3 super cell to build neighbour 
    information for     all atoms in the moire unit cell.

    Args:
        atom_pstn_list (np.ndarray): atom postions in a moire unit cell
        m_basis_vecs (dict) : moire basis vectors dictionary
        distance (float): KDtree searching cutoff distance. Defaults to 2.5113*A_C.

    Returns:
        tuple: all nn indices, 3x3 supercell postions 
    """

    m_unitvec_1 = m_basis_vecs['mu1']
    m_unitvec_2 = m_basis_vecs['mu2']

    # number of all atoms
    num_atoms = atom_pstn_list.shape[0]
    # add information for `d`
    m_unitvec_1 = np.array([m_unitvec_1[0], m_unitvec_1[1], 0])
    m_unitvec_2 = np.array([m_unitvec_2[0], m_unitvec_2[1], 0])

    # 3x3 supercell to perform KD tree searching scheme
    area1 = atom_pstn_list+m_unitvec_1
    area2 = atom_pstn_list+m_unitvec_2
    area3 = atom_pstn_list-m_unitvec_1
    area4 = atom_pstn_list-m_unitvec_2
    area5 = atom_pstn_list+m_unitvec_1+m_unitvec_2
    area6 = atom_pstn_list+m_unitvec_1-m_unitvec_2
    area7 = atom_pstn_list-m_unitvec_1+m_unitvec_2
    area8 = atom_pstn_list-m_unitvec_1-m_unitvec_2

    enlarge_atom_pstn_list = np.concatenate((atom_pstn_list, area1, area2, area3, area4, area5, area6, area7, area8))

    # kdtree search, only use first 2D information
    x = enlarge_atom_pstn_list[:, :2]
    y = atom_pstn_list[:, :2]
    tree = KDTree(x)
    ind = tree.query_radius(y, r=distance)

    # the kdtree algotithm provided by sklearn will return the index
    # including itself, the following code will remove them
    all_nns = np.array(
        [np.array([idx for idx in nn_indices if idx != i]) for i, nn_indices in enumerate(ind)],
        dtype=object,
    )

    return (all_nns, enlarge_atom_pstn_list)


def set_relative_dis_ndarray(atom_pstn_list: np.ndarray, enlarge_atom_pstn_list: np.ndarray,
                             all_nns: np.ndarray) -> tuple:
    """set up relative distance betweenn one atom and another

    Args:
        atom_pstn_list (np.ndarray): atom postions in a moire unit cell
        enlarge_atom_pstn_list (np.ndarray): atom postions in a 3x3 moire supercell
        all_nns (np.ndarray): nearest neighbor array

    Returns:
        tuple: (npair_dict, ndist_dict)
    """

    # print("num of atoms (code in moire set up):", len(atom_pstn_list))
    num_atoms = atom_pstn_list.shape[0]
    # tricky code here
    # construct Ri list
    neighbour_len_list = [subindex.shape[0] for subindex in all_nns]
    atom_pstn_2darray = np.repeat(atom_pstn_list, neighbour_len_list, axis=0)

    # construct Rj near Ri list
    atom_neighbour_2darray = enlarge_atom_pstn_list[np.hstack(all_nns)]
    assert atom_pstn_2darray.shape == atom_neighbour_2darray.shape

    ind = all_nns % num_atoms
    # ind = [np.sort(subind) for subind in ind]
    # (row, col) <=> (index_i, index_j)
    row = [iatom for iatom in range(num_atoms) for n in range(ind[iatom].shape[0])]
    col = [jatom for subindex in ind for jatom in subindex]
    # neighbour pair dict
    npair_dict = {}
    npair_dict['r'] = row
    npair_dict['c'] = col

    # first two dimentions (dri -drj)
    dr = (atom_pstn_2darray-atom_neighbour_2darray)[:, :2]
    # the thrid dimention  (dri -drj)
    dd = (atom_pstn_2darray-atom_neighbour_2darray)[:, -1]
    # neighour distance dict
    ndist_dict = {}
    ndist_dict['dr'] = dr
    ndist_dict['dd'] = dd

    return (npair_dict, ndist_dict)
