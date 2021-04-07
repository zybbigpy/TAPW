import numpy as np
import magnetic.moire_magnetic_setup as magset 

from scipy.linalg import block_diag
from scipy import sparse
from itertools import product


# eV
VPI_0 = -2.81
# eV
VSIGMA_0 = 0.48
# flux quantum (SI unit in Wb)
FLUX_QUANTUM = 2.067833848e-15

R_RANGE = 0.184*magset.A_0 


def _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g: int)->list:

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


def _set_kmesh(mm_g_unitvec_1, mm_g_unitvec_2, n_k: int, q: int)->list:
    
    # attention here, not normalized sampling
    k_step = 1/n_k
    kmesh = [i*k_step*mm_g_unitvec_1 + j*k_step*mm_g_unitvec_2 
             for (i, j) in product(range(n_k), range(n_k))]

    return kmesh


def _set_kmesh_disp(mm_g_unitvec_1, n_k: int):

    k_step = 1/n_k
    kmesh = [i*k_step*mm_g_unitvec_1 for i in range(n_k)]

    return kmesh


def _set_sk_integral(atom_pstn_2darray, atom_neighbour_2darray):
    """
    dr (*, 2) ndarray, dd (*, ) ndarray, * represents for neighbour pair

    -------
    Return:
    
    hopping: (*, ) ndarray
    """
    
    dr = (atom_pstn_2darray-atom_neighbour_2darray)[:,:2]
    dd = (atom_pstn_2darray-atom_neighbour_2darray)[:,-1]
    res = np.sum(dr**2, axis=1) + dd**2
    res_sqrt = np.sqrt(res)

    vpi = VPI_0*np.exp(-(res_sqrt-magset.A_EDGE)/R_RANGE)
    vsigma = VSIGMA_0*np.exp(-(res_sqrt-magset.D_AB)/R_RANGE)

    hopping = vpi*(1-dd**2/res)+vsigma*(dd**2)/res

    return hopping


def _set_ab_phase(atom_pstn_2darray, atom_neighbour_2darray, mag):
    """
    This function is used to calculate the Peierls phase when we add magnetic field
    
    ----------
    Parameters:

    atom_pstn_2darray, atom neighbour_2darray <==> (Ri, Rj), both are (*, 3) shape
    mag: magnetic field `B`
    """

    # B(x2+x1)(y2-y1)/2
    ab_phase = (mag/2)*(atom_neighbour_2darray[:,0]+atom_pstn_2darray[:,0])*(atom_neighbour_2darray[:,1]-atom_pstn_2darray[:,1])
    ab_phase = np.exp(-2j*np.pi*ab_phase)

    return ab_phase


def _set_const_mtrx(n_moire, m_g_unitvec_1,  m_g_unitvec_2, row, col, mm_atom_list,
                    atom_pstn_2darray, atom_neighbour_2darray, g_vec_list,  valley, mag):
    """
    calculate two constant matrix

    --------
    Returns:
    
    1. gr_mtrx (4*n_g, n_atom) np.matrix

    2. tr_mtrx (n_atom, n_atom) sparse matrix
    """

    n_g = len(g_vec_list)
    n_atom = len(mm_atom_list)
    
    factor = 1/np.sqrt(n_atom/4)
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g + valley*offset, r[:2]))
                        for g in g_vec_list for r in mm_atom_list]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

    hopping = _set_sk_integral(atom_pstn_2darray, atom_neighbour_2darray)
    abphase = _set_ab_phase(atom_pstn_2darray, atom_neighbour_2darray, mag)

    hopping_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))
    abphase_mtrx = sparse.csr_matrix((abphase, (row, col)), shape=(n_atom, n_atom))

    tr_mtrx = abphase_mtrx.multiply(hopping_mtrx)

    return (gr_mtrx, tr_mtrx)


def _cal_hamiltonian_k(atom_pstn_2darray, atom_neighbour_2darray, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom):
    
    dr = (atom_pstn_2darray-atom_neighbour_2darray)[:,:2]
    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))
    hr_mtrx = kr_mtrx.multiply(tr_mtrx)

    hamiltonian_k = gr_mtrx * (hr_mtrx * gr_mtrx.H)

    return hamiltonian_k


def mag_tb_solver(n_moire: int, n_g: int, n_k: int, valley: int, p: int, q: int, disp=False):
    """
    A wrapper,  the magnetic tightbinding solver

    -------
    Returns:

    (emesh, dmesh)
    emesh: eigenvalues, dnmesh: eigen vectors
    """
    
    (m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
     mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magset._set_moire_magnetic(n_moire, q)

    mag = p/(q*s)

    (mm_atom_list, enlarge_mm_atom_list) = magset.set_magnetic_atom_pstn(n_moire, q, "../data/")
    ind = magset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)

    (atom_pstn_2darray, atom_neighbour_2darray, row, col) = magset.set_relative_dis_ndarray(mm_atom_list, enlarge_mm_atom_list, ind)

    if disp:
        kmesh = _set_kmesh_disp(mm_g_unitvec_1, n_k)
    else:
        kmesh = _set_kmesh(mm_g_unitvec_1, mm_g_unitvec_2, n_k, q)
    
    g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g)
    (gr_mtrx, tr_mtrx) = _set_const_mtrx(n_moire, m_g_unitvec_1, m_g_unitvec_2, row, col, mm_atom_list, 
                                         atom_pstn_2darray, atom_neighbour_2darray, g_vec_list, valley, mag)

    n_atom = len(mm_atom_list)
    n_band = len(g_vec_list)*4
    n_kpts = len(kmesh)
    mag_tesla = p*FLUX_QUANTUM*(10**20)/(q*s)
    
    print('='*100)
    np.set_printoptions(6)
    print("magnetic field (T)".ljust(35),":", mag_tesla, "(", "p =", p, ", q =", q, ")")
    print("n moire is".ljust(35), ":", n_moire)
    print("valley  is".ljust(35), ":", valley)
    print("num of g vectors is".ljust(35), ":", n_g)
    print("num of atoms in magnetic lattice".ljust(35), ":", n_atom) 
    print("num of kpoints".ljust(35), ":", n_kpts)
    print("num of bands".ljust(35), ":", n_band)
    print('='*100)

    dmesh = []
    emesh = []
    count = 1

    for kvec in kmesh:
        print("in k sampling process count =", count)
        count += 1
        hamk = _cal_hamiltonian_k(atom_pstn_2darray, atom_neighbour_2darray, kvec, gr_mtrx, tr_mtrx, row, col, n_atom)
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        #print(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    
    print("k sampling process finished.")
    return (emesh, dmesh)
