import numpy as np
import magnetic.moire_magnetic_setup as magset 

from scipy.linalg import block_diag
from scipy import sparse
from itertools import product


# eV
VPI_0 = -2.7
# eV
VSIGMA_0 = 0.48
# flux quantum (SI unit in Wb) h/e
FLUX_QUANTUM = 2*2.067833848e-15
# tb hopping cut off 
R_RANGE = 0.184*magset.A_0 


def _set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int, q:int):

    g_vec_list=[]
    g_1 = m_g_unitvec_1
    g_2 = m_g_unitvec_2/q
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2
    start_pnt = (-m_g_unitvec_1-m_g_unitvec_2)*(n_g-1) + offset*valley

    for (i, j) in product(range(2*q*n_g), range(2*n_g)):
        g_vec_list.append(i*g_2 + j*g_1)

    g_vec_list = np.array(g_vec_list) + start_pnt

    return g_vec_list


def _set_kmesh(mm_g_unitvec_1, mm_g_unitvec_2, n_k:int, q:int)->list:
    
    # attention here, not normalized sampling
    k_step = 1/n_k
    kmesh = [i*k_step*mm_g_unitvec_1 + j*k_step*mm_g_unitvec_2 
             for (i, j) in product(range(n_k), range(n_k))]

    return kmesh


def _set_kmesh_disp(mm_g_unitvec_1, n_k:int):

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
    ab_phase = np.exp(2j*np.pi*ab_phase)

    return ab_phase


def _set_ab_phase_periodic(num_pairs, B, atom_pstn_2darray_frac, atom_neighbour_2darray_frac):
    import magnetic.periodic_guage as pg

    ab_phase = pg.set_ab_phase_list(num_pairs, B, atom_pstn_2darray_frac, atom_neighbour_2darray_frac)
    ab_phase = np.exp(2j*np.pi*ab_phase)

    return ab_phase


def _set_const_mtrx_periodic1(n_moire, m_g_unitvec_1,  m_g_unitvec_2, row, col, mm_atom_list,
                    atom_pstn_2darray, atom_neighbour_2darray, g_vec_list, valley, mag):
    """
    calculate two constant matrix in periodic guage

    --------
    Returns:
    
    1. gr_mtrx (4*n_g, n_atom) np.matrix

    2. tr_mtrx (n_atom, n_atom) sparse matrix
    """

    n_g = len(g_vec_list)
    n_atom = len(mm_atom_list)
    n_pairs = atom_pstn_2darray.shape[0]

    factor = 1/np.sqrt(n_atom/4)
   
    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2]))
                        for g in g_vec_list for r in mm_atom_list]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

    (atom_pstn_2darray_frac, atom_neighbour_2darray_frac) = magset.set_frac_coordinate1(atom_pstn_2darray, atom_neighbour_2darray, 
                                                                                       m_g_unitvec_1,     m_g_unitvec_2)
    hopping = _set_sk_integral(atom_pstn_2darray, atom_neighbour_2darray)
    abphase = _set_ab_phase_periodic(n_pairs, mag, atom_pstn_2darray_frac, atom_neighbour_2darray_frac)

    hopping_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))
    hopping_mtrx_cc = (hopping_mtrx.transpose()).conjugate()
    hopping_mtrx_delta = hopping_mtrx - hopping_mtrx_cc

    if hopping_mtrx_delta.max()>1.0E-9:
        print(hopping_mtrx_delta.max())
        raise Exception("hopping matrix is not hermitian?!") 
    

    abphase_mtrx = sparse.csr_matrix((abphase, (row, col)), shape=(n_atom, n_atom))
    abphase_mtrx_cc = (abphase_mtrx.transpose()).conjugate()
    abphase_mtrx_delta = abphase_mtrx - abphase_mtrx_cc

    if abphase_mtrx_delta.max()>1.0E-10:
        print(abphase_mtrx_delta.max())
        raise Exception("abphase matrix is not hermitian?!")   
    
    tr_mtrx = abphase_mtrx.multiply(hopping_mtrx)

    return (gr_mtrx, tr_mtrx)


def _set_ab_phase_periodic2(num_pairs, B, atom_pstn_2darray_frac, atom_neighbour_2darray_frac):

    atom_neighbour_2darray_frac_floor = np.floor(atom_neighbour_2darray_frac)
    atom_pstn_2darray_frac_floor = np.floor(atom_pstn_2darray_frac)

    ab_phase = _set_ab_phase(atom_pstn_2darray_frac_floor, atom_neighbour_2darray_frac_floor, B)
    print("use landau gauge")
    return ab_phase


def _set_const_mtrx_periodic2(n_moire, m_g_unitvec_1,  m_g_unitvec_2, row, col, mm_atom_list,
                    atom_pstn_2darray, atom_neighbour_2darray, g_vec_list, valley, mag):
    """
    calculate two constant matrix in periodic guage

    --------
    Returns:
    
    1. gr_mtrx (4*n_g, n_atom) np.matrix

    2. tr_mtrx (n_atom, n_atom) sparse matrix
    """

    n_g = len(g_vec_list)
    n_atom = len(mm_atom_list)
    n_pairs = atom_pstn_2darray.shape[0]

    factor = 1/np.sqrt(n_atom/4)
   
    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2]))
                        for g in g_vec_list for r in mm_atom_list]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

    (atom_pstn_2darray_frac, atom_neighbour_2darray_frac) = magset.set_frac_coordinate2(atom_pstn_2darray, atom_neighbour_2darray, 
                                                                                       m_g_unitvec_1,     m_g_unitvec_2)
    hopping = _set_sk_integral(atom_pstn_2darray, atom_neighbour_2darray)
    abphase = _set_ab_phase_periodic2(n_pairs, mag, atom_pstn_2darray_frac, atom_neighbour_2darray_frac)

    hopping_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))
    hopping_mtrx_cc = (hopping_mtrx.transpose()).conjugate()
    hopping_mtrx_delta = hopping_mtrx - hopping_mtrx_cc

    if hopping_mtrx_delta.max()>1.0E-9:
        print(hopping_mtrx_delta.max())
        raise Exception("hopping matrix is not hermitian?!") 
    
    abphase_mtrx = sparse.csr_matrix((abphase, (row, col)), shape=(n_atom, n_atom))
    abphase_mtrx_cc = (abphase_mtrx.transpose()).conjugate()
    abphase_mtrx_delta = abphase_mtrx - abphase_mtrx_cc 

    if abphase_mtrx_delta.max()>1.0E-10:
        print(abphase_mtrx_delta.max())
        raise Exception("abphase matrix is not hermitian?!")   
    
    tr_mtrx = abphase_mtrx.multiply(hopping_mtrx)
    tr_mtrx_cc = (tr_mtrx.transpose()).conjugate()
    tr_mtrx_delta = tr_mtrx - tr_mtrx_cc

    if tr_mtrx_delta.max()>1.0E-9:
        print(tr_mtrx_delta.max())
        raise Exception("tr matrix is not hermitian?!")

    return (gr_mtrx, tr_mtrx)


def _cal_hamiltonian_k(atom_pstn_2darray, atom_neighbour_2darray, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom):
    
    dr = (atom_pstn_2darray-atom_neighbour_2darray)[:,:2]
    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))
    
    kr_mtrx_cc = (kr_mtrx.transpose()).conjugate()
    kr_mtrx_delta = kr_mtrx - kr_mtrx_cc

    if kr_mtrx_delta.max()>1.0E-10:
        print(kr_mtrx_delta.max())
        raise Exception("kr matrix is not hermitian?!")

    hr_mtrx = kr_mtrx.multiply(tr_mtrx)
    hr_mtrx_cc = (hr_mtrx.transpose()).conjugate()
    hr_mtrx_delta = hr_mtrx - hr_mtrx_cc

    if hr_mtrx_delta.max()>1.0E-10:
        print(hr_mtrx_delta.max())
        raise Exception("hr matrix is not hermitian?!")
    
    hamk = gr_mtrx * (hr_mtrx * gr_mtrx.H)
    hamk_cc = (hamk.transpose()).conjugate()
    hamk_delta = hamk - hamk_cc

    if hamk_delta.max()>1.0E-9:
        print(hamk_delta.max())
        raise Exception("hamk matrix is not hermitian?!")

    err  = np.linalg.norm(hamk-hamk.T.conj())/np.linalg.norm(hamk+hamk.T.conj())
    print("error in hamk construction is:", err)

    return hamk


def mag_tb_solver_periodic(n_moire:int, n_g:int, n_k:int, valley:int, p:int, q:int, type=1, disp=False):
    """
    the magnetic tightbinding solver in periodic guage

    -------
    Returns:

    (emesh, dmesh)
    emesh: eigenvalues, dnmesh: eigen vectors
    """
    
    (m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
     mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magset._set_moire_magnetic(n_moire, q)

    mag = p/q

    ###############################################################
    # test use, check whether return to non mag case when p=0,q=1
    # kline = 0
    # m_gamma_vec = np.array([0, 0])
    # m_k1_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_2/3
    # m_k2_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_1/3
    # m_m_vec = (m_k1_vec + m_k2_vec)/2
    ###############################################################

    (mm_atom_list, enlarge_mm_atom_list) = magset.set_magnetic_atom_pstn(n_moire, q, "../data/")
    ind = magset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)

    (atom_pstn_2darray, atom_neighbour_2darray, row, col) = magset.set_relative_dis_ndarray(mm_atom_list, enlarge_mm_atom_list, ind)

    if disp:
        kmesh = _set_kmesh_disp(mm_g_unitvec_1, n_k)
        # test use, check whether return to non mag case when p=0,q=1
        # (kline, kmesh) = _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, n_k)
    else:
        kmesh = _set_kmesh(mm_g_unitvec_1, mm_g_unitvec_2, n_k, q)
    
    g_vec_list = _set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley, q)

    if type == 2: ## something wrong, may not use
        (gr_mtrx, tr_mtrx) = _set_const_mtrx_periodic2(n_moire, m_g_unitvec_1, m_g_unitvec_2, row, col, mm_atom_list, 
                                         atom_pstn_2darray, atom_neighbour_2darray, g_vec_list, valley, mag)
    elif type == 1:
        (gr_mtrx, tr_mtrx) = _set_const_mtrx_periodic1(n_moire, m_g_unitvec_1, m_g_unitvec_2, row, col, mm_atom_list, 
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

    # dmesh = []
    # emesh = []
    # count = 1

    # for kvec in kmesh:
    #     print("in k sampling process count =", count)
    #     count += 1
    #     hamk = _cal_hamiltonian_k(atom_pstn_2darray, atom_neighbour_2darray, kvec, gr_mtrx, tr_mtrx, row, col, n_atom)
    #     eigen_val, eigen_vec = np.linalg.eigh(hamk)
    #     #print("hamk min val:", np.min(hamk))
    #     #print("hamk max err:", np.max(hamk-hamk.T.conj()))
    #     #print(eigen_val)
    #     emesh.append(eigen_val)
    #     dmesh.append(eigen_vec)
    
    # print("k sampling process finished.")
    hamk = _cal_hamiltonian_k(atom_pstn_2darray, atom_neighbour_2darray, np.array([0,0]), gr_mtrx, tr_mtrx, row, col, n_atom)
    eigen_val, eigen_vec = np.linalg.eigh(hamk)

    return (eigen_val, eigen_vec)


def mag_tb_project(n_moire:int, n_g:int, n_k:int, valley:int, p:int, q:int):
    """
    project magnetic Hamiltonian on 2q flat bands, degenerate perturbation approximation
    """

    (m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
     mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magset._set_moire_magnetic(n_moire, q)

    mag = p/(q*s)
    mag_tesla = p*FLUX_QUANTUM*(10**20)/(q*s)
    print("(p, q):", (p, q), "mag field (T):", mag_tesla)
    kmesh = _set_kmesh_disp(mm_g_unitvec_1, n_k)

    (mm_atom_list, enlarge_mm_atom_list) = magset.set_magnetic_atom_pstn(n_moire, q, "../data/")
    ind = magset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)
    (atom_pstn_2darray, atom_neighbour_2darray, row, col) = magset.set_relative_dis_ndarray(mm_atom_list, enlarge_mm_atom_list, ind)


    g_vec_list = _set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley, q)
    (gr_mtrx, tr_mtrx) = _set_const_mtrx_periodic2(n_moire, m_g_unitvec_1, m_g_unitvec_2, row, col, mm_atom_list, 
                                         atom_pstn_2darray, atom_neighbour_2darray, g_vec_list, valley, mag)


    n_kpts = len(kmesh)
    n_atom = len(mm_atom_list)
    dmesh_name = "../data/dmesh_n30_p0_q"+str(q)+"_v"+str(valley)+".npy"
    dmesh_load = np.load(dmesh_name)
    print("The dmesh loaded shape:", dmesh_name, dmesh_load.shape)

    emesh = []
    emesh_proj = []

    for i in range(n_kpts):
        hamk = _cal_hamiltonian_k(atom_pstn_2darray, atom_neighbour_2darray, 
                                  kmesh[i],  gr_mtrx, tr_mtrx, row, col, n_atom)
        wffunc = dmesh_load[i]
        # 2q flat bands
        ind1 = wffunc.shape[0]//2-q
        ind2 = wffunc.shape[0]//2+q

        # print("ind1, ind2:", ind1, ind2)
        # print("hamk shape:", hamk.shape)
        # print("func shape:", func_proj.shape)
        # print("hamk min val:", np.min(hamk))
        # print("hamk max err:", np.max(hamk-hamk.T.conj()))

        #func_proj = wffunc[:, ind1:ind2]
        func_proj = wffunc[:, :]
        
        if np.max(hamk-hamk.T.conj())>1.0E-8:
            raise Exception("Hamiltonian matrix is not hermitian?!")
        
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        
        emesh.append(eigen_val[ind1:ind2])
        hamk_proj = (np.conjugate(np.transpose(func_proj)))@hamk@(func_proj)

        if np.max(hamk_proj-hamk_proj.T.conj())>1.0E-8:
            raise Exception("Hamiltonian matrix is not hermitian?!")        
        # print("hamk proj min val:", np.min(hamk_proj))
        # print("hamk proj max err:", np.max(hamk_proj-hamk_proj.T.conj())) 

        eigen_val_proj, eigen_vec_proj = np.linalg.eigh(hamk_proj)
        emesh_proj.append(eigen_val_proj[ind1:ind2])
        print("eigen val hamk full:", eigen_val[ind1:ind2])
        print("eigen val hamk proj:", eigen_val_proj[ind1:ind2])
        # print("absolute err  (meV):", 1000*(eigen_val_proj-eigen_val[ind1:ind2]))
        # print("relative err (100%):",  100*(eigen_val_proj-eigen_val[ind1:ind2])/eigen_val[ind1:ind2])
        print("="*100)

    return (np.array(emesh), np.array(emesh_proj))


#   def _set_const_mtrx(n_moire, m_g_unitvec_1,  m_g_unitvec_2, row, col, mm_atom_list,
#                     atom_pstn_2darray, atom_neighbour_2darray, g_vec_list, valley, mag):
#     """
#     calculate two constant matrix

#     --------
#     Returns:
    
#     1. gr_mtrx (4*n_g, n_atom) np.matrix

#     2. tr_mtrx (n_atom, n_atom) sparse matrix
#     """

#     n_g = len(g_vec_list)
#     n_atom = len(mm_atom_list)
    
#     factor = 1/np.sqrt(n_atom/4)
#     #offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

#     gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2]))
#                         for g in g_vec_list for r in mm_atom_list]).reshape(n_g, n_atom)

#     g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
#     gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

#     hopping = _set_sk_integral(atom_pstn_2darray, atom_neighbour_2darray)
#     abphase = _set_ab_phase(atom_pstn_2darray, atom_neighbour_2darray, mag)

#     hopping_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))
#     hopping_mtrx_cc = (hopping_mtrx.transpose()).conjugate()
#     hopping_mtrx_delta = hopping_mtrx - hopping_mtrx_cc

#     if hopping_mtrx_delta.max()>1.0E-9:
#         print(hopping_mtrx_delta.max())
#         raise Exception("hopping matrix is not hermitian?!") 
    
#     abphase_mtrx = sparse.csr_matrix((abphase, (row, col)), shape=(n_atom, n_atom))   
#     tr_mtrx = abphase_mtrx.multiply(hopping_mtrx)

#     return (gr_mtrx, tr_mtrx)

def mag_tb_solver_test(n_moire:int, n_g:int, n_k:int, valley:int, p:int, q:int):
    """
    the magnetic tightbinding solver (test use)

    -------
    Returns:

    (emesh, dmesh)
    emesh: eigenvalues, dnmesh: eigen vectors
    """
    
    (m_unitvec_1,    m_unitvec_2,  m_g_unitvec_1, m_g_unitvec_2, 
     mm_unitvec_1, mm_unitvec_2,   mm_g_unitvec_1, mm_g_unitvec_2, s) = magset._set_moire_magnetic(n_moire, q)

    mag = p/(q*s)

    ###############################################################
    # test use, check whether return to non mag case when p=0,q=1
    kline = 0
    m_gamma_vec = np.array([0, 0])
    m_k1_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_2/3
    m_k2_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_1/3
    m_m_vec = (m_k1_vec + m_k2_vec)/2
    ###############################################################

    (mm_atom_list, enlarge_mm_atom_list) = magset.set_magnetic_atom_pstn(n_moire, q, "../data/")
    ind = magset.set_magnetic_atom_neighbour_list(mm_atom_list, enlarge_mm_atom_list)

    (atom_pstn_2darray, atom_neighbour_2darray, row, col) = magset.set_relative_dis_ndarray(mm_atom_list, enlarge_mm_atom_list, ind)


    (kline, kmesh) = _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, n_k)

    
    g_vec_list = _set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley, q)
    (gr_mtrx, tr_mtrx) = _set_const_mtrx_periodic1(n_moire, m_g_unitvec_1, m_g_unitvec_2, row, col, mm_atom_list, 
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

        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    
    print("k sampling process finished.")
    return (np.array(emesh), np.array(dmesh), kline)

# test use, check whether return to non mag case when p = 0, q = 0
def _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, nk):
    """
    moire dispertion, this code is just modifield from Prof Dai's realization
    """

    num_sec = 4
    ksec = np.zeros((num_sec,2),  float)
    num_kpt = nk * (num_sec - 1)
    kline = np.zeros((num_kpt),  float)
    kmesh = np.zeros((num_kpt,2),float)

    # set k path (K1 - Gamma - M - K2)
    ksec[0] = m_k1_vec
    ksec[1] = m_gamma_vec
    ksec[2] = m_m_vec
    ksec[3] = m_k2_vec

    for i in range(num_sec-1):
        vec = ksec[i+1] - ksec[i]
        klen = np.sqrt(np.dot(vec,vec))
        step = klen/(nk)

        for ikpt in range(nk):
            kline[ikpt+i*nk] = kline[i*nk-1] + ikpt * step   
            kmesh[ikpt+i*nk] = vec*ikpt/(nk-1) + ksec[i]

    return (kline, kmesh)