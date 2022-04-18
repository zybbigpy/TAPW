import numpy as np
import tightbinding.moire_setup as mset
import matplotlib.pyplot as plt
import scipy.linalg as la

from scipy.linalg import block_diag
from scipy import sparse
from itertools import product

# eV
VPI_0 = -2.7
# eV
VSIGMA_0 = 0.48

R_RANGE = 0.184*mset.A_0


def _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int):
    """
    old version code, aborted.
    """
    g_vec_list = []

    # construct a hexagon area by using three smallest g vectors (with symmetry)
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    # for (i, j) in product(range(n_g), range(n_g)):
    #     g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    # for (i, j) in product(range(1, n_g), range(1, n_g)):
    #     g_vec_list.append(i*g_3 + j*m_g_unitvec_1)
    
    # for i in range(n_g):
    #     for j in range(1, n_g):
    #         g_vec_list.append(j*g_3 + i*m_g_unitvec_2)

    for i in range(n_g):
        for j in range(n_g):
            g_vec_list.append(i*m_g_unitvec_1+j*m_g_unitvec_2)
    
    for i in range(n_g):
        for j in range(1, n_g):
            g_vec_list.append(-j*m_g_unitvec_1+(i-j)*m_g_unitvec_2)

    for i in range(1, n_g):
        for j in range(1, n_g):
            g_vec_list.append(-i*m_g_unitvec_2+(j-i)*m_g_unitvec_1)
    
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2
     
    return np.array(g_vec_list)+offset*valley


def _set_g_vec_list_symm(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int):

    g_vec_list = []
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

    # construct a hexagon area by using three smallest g vectors (with symmetry)
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*g_3 + j*m_g_unitvec_1)
    
    for (i, j) in product(range(n_g), range(n_g)):
            g_vec_list.append(j*g_3 + i*m_g_unitvec_2)

    # remove repeated gvecs in glist
    g_vec_list = np.unique(np.array(g_vec_list), axis=0) + offset*valley

    return g_vec_list


def _set_g_vec_list_nsymm(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int):

    g_vec_list = []
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

    g_1 = m_g_unitvec_1
    g_2 = m_g_unitvec_2
    g_3 = -m_g_unitvec_1
    g_4 = -m_g_unitvec_2

    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*g_1 + j*g_2)

    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*g_1 + j*g_4)

    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*g_2 + j* g_3)

    for (i, j) in product(range(n_g), range(n_g)):
        g_vec_list.append(i*g_3 + j*g_4)
    

    print("G list shape before unique:", len(g_vec_list))

    g_vec_list = np.unique(np.array(g_vec_list), axis=0) + offset*valley

    print("G list shape after unique:", g_vec_list.shape)

    return g_vec_list


def _set_g_vec_list_nsymm_2(m_g_unitvec_1, m_g_unitvec_2, n_g:int, n_moire:int, valley:int):

    g_vec_list = []
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2
    start_pnt = g_3*(n_g-1) + offset*valley

    for (i, j) in product(range(2*n_g), range(2*n_g)):
        g_vec_list.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    g_vec_list = np.array(g_vec_list) + start_pnt

    return g_vec_list


def _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k:int)->list:
    
    k_step = 1/n_k
    kmesh = [i*k_step*m_g_unitvec_1 + j*k_step*m_g_unitvec_2 
             for (i, j) in product(range(n_k), range(n_k))]

    return kmesh


def _sk_integral(dr, dd):
    """
    dr (*, 2) ndarray, dd (*, ) ndarray, * represents for interaction pair

    -------
    Return:
    
    hopping: (*, ) ndarray
    """
    res = np.sum(dr**2, axis=1) + dd**2
    res_sqrt = np.sqrt(res)

    vpi = VPI_0*np.exp(-(res_sqrt-mset.A_EDGE)/R_RANGE)
    vsigma = VSIGMA_0*np.exp(-(res_sqrt-mset.D_AB)/R_RANGE)

    hopping = vpi*(1-dd**2/res)+vsigma*(dd**2)/res

    return hopping


def _set_const_mtrx(n_moire,  dr,  dd,    m_g_unitvec_1,  m_g_unitvec_2, 
                    row, col, g_vec_list, atom_pstn_list, valley):
    """
    calculate two constant matrix

    --------
    Returns:
    
    1. gr_mtrx (4*n_g, n_atom) np.matrix

    2. tr_mtrx (n_atom, n_atom) sparse matrix
    """

    n_g = len(g_vec_list)
    n_atom = len(atom_pstn_list)
    
    factor = 1/np.sqrt(n_atom/4)
    offset = n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2

    # in old version code, offset of Glist realized here
    # gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g + valley*offset, r[:2]))
    #                     for g in g_vec_list for r in atom_pstn_list]).reshape(n_g, n_atom)

    # new version code, in g list construction

    gr_mtrx = np.array([factor*np.exp(-1j*np.dot(g, r[:2]))
                        for g in g_vec_list for r in atom_pstn_list]).reshape(n_g, n_atom)

    g1, g2, g3, g4 = np.hsplit(gr_mtrx, 4)
    gr_mtrx = np.matrix(block_diag(g1, g2, g3, g4))

    hopping = _sk_integral(dr, dd)
    tr_mtrx = sparse.csr_matrix((hopping, (row, col)), shape=(n_atom, n_atom))

    tr_mtrx_cc = (tr_mtrx.transpose()).conjugate()
    tr_mtrx_delta = tr_mtrx - tr_mtrx_cc
    
    if tr_mtrx_delta.max()>1.0E-9:
        print(tr_mtrx_delta.max())
        raise Exception("Hamiltonian matrix is not hermitian?!")     

    return (gr_mtrx, tr_mtrx)


def _cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom):
    
    tk_data = np.exp(-1j*np.dot(dr, k_vec))
    kr_mtrx = sparse.csr_matrix((tk_data, (row, col)), shape=(n_atom, n_atom))

    kr_mtrx_cc = (kr_mtrx.transpose()).conjugate()
    kr_mtrx_delta = kr_mtrx - kr_mtrx_cc

    if kr_mtrx_delta.max()>1.0E-9:
        print(kr_mtrx_delta.max())
        raise Exception("Hamiltonian matrix is not hermitian?!")  


    hr_mtrx = kr_mtrx.multiply(tr_mtrx)

    hr_mtrx_cc = (hr_mtrx.transpose()).conjugate()
    hr_mtrx_delta = hr_mtrx - hr_mtrx_cc
    
    if hr_mtrx_delta.max()>1.0E-9:
        print(hr_mtrx_delta.max())
        raise Exception("Hamiltonian matrix is not hermitian?!")  

    hamiltonian_k = gr_mtrx * (hr_mtrx * gr_mtrx.H)

    return hamiltonian_k


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


def _set_kmesh_neighbour(g_vec_list, n_k, m_g_unitvec_1, m_g_unitvec_2, n_moire, valley):
    
    offset = valley*(n_moire*m_g_unitvec_1 + n_moire*m_g_unitvec_2)
    g_vec_list = g_vec_list - offset
    print("check here!!!!!",g_vec_list[0])
    num_g = g_vec_list.shape[0]
    err = 0.02*np.dot(m_g_unitvec_1, m_g_unitvec_1)

    transmat_list = []
    for m in range(num_g):
        mat = np.zeros((num_g, num_g), float)
        q_vec = g_vec_list[m]
        for (i, j) in product(range(num_g), range(num_g)):
            diff_vec = g_vec_list[i] + q_vec - g_vec_list[j]
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
    
    return transmat_list, neighbor_map


def tightbinding_solver(n_moire:int, n_g:int, n_k:int, valley:int, disp=False, symm=True, relax=False)->tuple:
    """  
    Tight binding solver for moire system

    -------
    Returns:
    
    1. emesh: eigenvalues, np.array(n_k, n_bands)
    2. dmesh: eigenvectors, np.array(n_k, n_bands, n_bands)
    3. kline: 0 when uniform sampling in 1st B.Z., k path for tb disp
    """
    
    
    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
     m_k2_vec,      m_m_vec,     rt_mtrx_half) = mset._set_moire(n_moire)

    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    atom_pstn_list = mset.read_atom_pstn_list(n_moire, relax)

    # old solution 
    # atom_neighbour_list = mset.read_atom_neighbour_list("../data/", n_moire)
    # (dr, dd, row, col) = mset.set_relative_dis_ndarray(atom_pstn_list, atom_neighbour_list, m_g_unitvec_1, 
    #                                                     m_g_unitvec_2,  m_unitvec_1,         m_unitvec_2)

    # new solution
    (all_nns, enlarge_atom_pstn_list)= mset.set_atom_neighbour_list(atom_pstn_list, m_unitvec_1, m_unitvec_2)
    (dr, dd, row, col) = mset.set_relative_dis_ndarray_new(atom_pstn_list, enlarge_atom_pstn_list, all_nns)

    if(disp): # k-path sampling
        (kline, kmesh) = _set_tb_disp_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, n_k)
    else:     # uniform sampling
        kmesh = _set_kmesh(m_g_unitvec_1, m_g_unitvec_2, n_k)
    
    
    #g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g)
    # symmetry G list or non symmetry
    # if symm:
    #     g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley)
    # else:
    #     print("nsymm2 g list construction.")
    #     g_vec_list = _set_g_vec_list_nsymm_2(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley)

    g_vec_list = _set_g_vec_list(m_g_unitvec_1, m_g_unitvec_2, n_g, n_moire, valley)
    gr_mtrx, tr_mtrx = _set_const_mtrx(n_moire,  dr,    dd,  m_g_unitvec_1,  m_g_unitvec_2, 
                                       row, col, g_vec_list, atom_pstn_list, valley)
    
    transmat_list, neighbor_map = _set_kmesh_neighbour(g_vec_list, n_k, m_g_unitvec_1, m_g_unitvec_2, n_moire, valley)

    n_atom = len(atom_pstn_list)
    n_band = len(g_vec_list)*4
    n_kpts = len(kmesh)

    print('='*100)
    np.set_printoptions(6)
    print("atom unit vector".ljust(30), ":", mset.A_UNITVEC_1, mset.A_UNITVEC_2)
    print("atom reciprotocal unit vector".ljust(30), ":", mset.A_G_UNITVEC_1, mset.A_G_UNITVEC_2)
    print("moire unit vector".ljust(30), ":", m_unitvec_1, m_unitvec_2)
    print("moire recoprotocal unit vector".ljust(30), ":", m_g_unitvec_1, m_g_unitvec_2)
    print("num of atoms".ljust(30), ":", n_atom) 
    print("num of kpoints".ljust(30), ":", n_kpts)
    print("num of bands".ljust(30), ":", n_band)
    print('='*100)

    for k_vec in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = _cal_hamiltonian_k(dr, k_vec, gr_mtrx, tr_mtrx, row, col, n_atom)
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        if np.max(eigen_val) > emax:
            emax = np.max(eigen_val)
        if np.min(eigen_val) < emin:
            emin = np.min(eigen_val)
        #print(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    print('='*100)
    print("emax =", emax, "emin =", emin)
    print('='*100)

    return (np.array(emesh), np.array(dmesh), kline, transmat_list, neighbor_map)


def tightbinding_plot(n_moire:int, n_g:int, n_k:int, band:int, symm:bool, name:str, relax:bool):

    emesh, dmesh, kline, _, _ = tightbinding_solver(n_moire, n_g, n_k, 1, True, symm, relax)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels([r'$\bar{K}$',  r'$\bar{\Gamma}$',  r'$\bar{M}$', r'$\bar{K}^\prime$'])
    ax.set_xlim(0, kline[-1])

    for i in range(band):
        plt.plot(kline, emesh[:, n_band//2+i],'-b')
        plt.plot(kline, emesh[:, n_band//2-1-i],'-b')

    #plt.plot(kline, emesh[:, n_band//2-1],'-b')
    #plt.plot(kline, emesh[:, n_band//2-2])

    emesh, dmesh, kline, _, _ = tightbinding_solver(n_moire, n_g, n_k, -1, True, symm, relax)

    for i in range(band):
        plt.plot(kline, emesh[:, n_band//2+i],'--r')
        plt.plot(kline, emesh[:, n_band//2-1-i],'--r')

    #plt.plot(kline, emesh[:, n_band//2-1],'--r')
    #plt.plot(kline, emesh[:, n_band//2-2])

    ax.set_ylabel("Engergy (eV)")
    ax.set_title("Tight Binding Band Structure of TBG"+" Nmoire "+str(n_moire))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    if relax:
        plt.savefig("../output/relaxbands"+str(n_moire)+name+".png", dpi=500)
    else:
        plt.savefig("../output/tbbands"+str(n_moire)+name+".png", dpi=500)


def index(x, y, n_k):
    return (x%n_k)*n_k+(y%n_k)


def d(x, y, n_k):

    dx, dy = 0, 0
    if x == n_k:
        dx = 1
    if y == n_k:
        dy = 1

    return (dx, dy)


def braket_norm(phi1, phi2, x1, y1, x2, y2, n_k, trans, nmap):

    dx1,dy1 = d(x1,y1,n_k)
    dx2,dy2 = d(x2,y2,n_k)
    amat1 = trans[nmap[0, dx1, dy1]]
    amat2 = trans[nmap[0, dx2, dy2]]
    braket = ((amat1.T)@phi1).transpose().conj().dot(amat2.T@phi2)
    res_det = la.det(braket)

    return res_det/la.norm(res_det)


def ux(bands, x, y, n_k, init, last, transmat_list, neighbor_map):
    phi1 = bands[index(x,   y, n_k)][:, init:last+1]
    phi2 = bands[index(x+1, y, n_k)][:, init:last+1]

    return braket_norm(phi1, phi2, x, y, x+1, y, n_k, transmat_list, neighbor_map)


def uy(bands, x, y, n_k, init, last, transmat_list, neighbor_map):
    phi1 = bands[index(x,   y, n_k)][:, init:last+1]
    phi2 = bands[index(x, y+1, n_k)][:, init:last+1]

    return braket_norm(phi1, phi2, x, y, x, y+1, n_k, transmat_list, neighbor_map)


def small_loop(bands, m, n, n_k, init, last, transmat_list, neighbor_map):

    return np.log(ux(bands, m,   n, n_k, init, last, transmat_list, neighbor_map)
                * uy(bands, m+1, n, n_k, init, last, transmat_list, neighbor_map)
                / ux(bands, m, n+1, n_k, init, last, transmat_list, neighbor_map)
                / uy(bands, m,   n, n_k, init, last, transmat_list, neighbor_map))


def cal_chern(bands, n_k, init, last, transmat_list, neighbor_map):

    ret = 0

    for m in range(n_k):
        for n in range(n_k):
            ret += small_loop(bands, m, n, n_k, init, last, transmat_list, neighbor_map)
        
    return ret/(2*np.pi*1j)