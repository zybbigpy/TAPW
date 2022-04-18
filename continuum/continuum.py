import numpy as np
import matplotlib.pyplot as plt

from itertools import product


"""
Reference: Continuum Model Derived By Koshino
1. PhysRevX.8.031087
2. New J. Phys. 17 015014
"""


# lattice constant
A0 = 2.46
# atomic reciprocal vector
A_G_UNITVEC_1 = np.array([2*np.pi/A0, -2*np.pi/(A0*np.sqrt(3))])
A_G_UNITVEC_2 = np.array([0,           4*np.pi/(A0*np.sqrt(3))])
# unit vector for atom system
A_UNITVEC_1 = np.array([A0,     0])
A_UNITVEC_2 = np.array([A0/2,  np.sqrt(3)*A0/2])
# fermi velocity 2.1354eV*a
HBARVF = 2.1354*A0
# two paramters, unit eV (chiral limit U1 = U2)
U1 = 0.0797
U2 = 0.0975
# pauli matrices
SIGMA_X = np.array([[0, 1],  [1, 0]])
SIGMA_Y = np.array([[0,-1j], [1j, 0]])
SIGMA_Z = np.array([[1, 0],  [0, -1]])


def _set_moire_angle(n_moire:int)->float:
    
    angle_r = np.arcsin(np.sqrt(3)*(2*n_moire+1)/(6*n_moire**2+6*n_moire+2))
    print("nmoire:", n_moire, ", equals angle(degree):", angle_r/np.pi*180)

    return angle_r


def _set_rt_mtrx(theta:float):

    rt_mtrx = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

    return rt_mtrx


def _set_moire(n_moire:int)->tuple:

    rt_angle = _set_moire_angle(n_moire)
    #rt_mtrx = _set_rt_mtrx(rt_angle)
    rt_mtrx_half = _set_rt_mtrx(rt_angle/2)

    # first `m_` represents for moire
    # moire unit vector
    m_unitvec_1 = (-n_moire*A_UNITVEC_1 + (2*n_moire +1)*A_UNITVEC_2)@rt_mtrx_half.T
    m_unitvec_2 = (-(2*n_moire+1)*A_UNITVEC_1 + (n_moire +1)*A_UNITVEC_2)@rt_mtrx_half.T
    
    # moire reciprocal vector
    m_g_unitvec_1 = A_G_UNITVEC_1@rt_mtrx_half.T - A_G_UNITVEC_1@rt_mtrx_half
    m_g_unitvec_2 = A_G_UNITVEC_2@rt_mtrx_half.T - A_G_UNITVEC_2@rt_mtrx_half
    
    # high symmetry points
    m_gamma_vec = np.array([0, 0])
    m_k1_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_2/3
    m_k2_vec = (m_g_unitvec_1 + m_g_unitvec_2)/3 + m_g_unitvec_1/3
    m_m_vec = (m_k1_vec + m_k2_vec)/2

    return (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
            m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
            m_k2_vec,      m_m_vec,     rt_mtrx_half)


def _set_kpt(rotmat):

    kpt1 = -(2*A_G_UNITVEC_1@rotmat.T+A_G_UNITVEC_2@rotmat.T)/3
    kpt2 = -(2*A_G_UNITVEC_1@rotmat  +A_G_UNITVEC_2@rotmat  )/3

    return (kpt1, kpt2)


def _check_eq(vec1, vec2):

    assert vec1.shape==vec2.shape
    
    if np.linalg.norm(vec1-vec2)<1E-9:
        return True
    else:
        return False


def _make_glist(n_g, n_moire, m_g_unitvec_1, m_g_unitvec_2, valley):
    
    glist = []
    offset = n_moire*m_g_unitvec_2

    # construct a hexagon area by using three smallest g vectors (with symmetry)
    g_3 = -m_g_unitvec_1-m_g_unitvec_2
    
    for (i, j) in product(range(n_g), range(n_g)):
        glist.append(i*m_g_unitvec_1 + j*m_g_unitvec_2)
    
    for (i, j) in product(range(n_g), range(n_g)):
        glist.append(i*g_3 + j*m_g_unitvec_1)
    
    for (i, j) in product(range(n_g), range(n_g)):
        glist.append(j*g_3 + i*m_g_unitvec_2)

    # remove repeated gvecs in glist
    glist = np.unique(np.array(glist), axis=0) + offset*valley

    return glist


def _make_transfer_const(m_g_unitvec_1, m_g_unitvec_2, valley):
    
    # three nearest g vec
    g1 = np.array([0, 0])
    g2 = valley*m_g_unitvec_1
    g3 = valley*(m_g_unitvec_1+m_g_unitvec_2)

    # three transfer matrix
    phase1 = np.exp(2j*np.pi/3)**valley
    phase2 = np.conj(phase1)
    t1 = np.array([[U1, U2],       [U2,        U1]])
    t2 = np.array([[U1, U2*phase2],[U2*phase1, U1]])
    t3 = np.array([[U1, U2*phase1],[U2*phase2, U1]])

    return (g1, g2, g3, t1, t2, t3)


def _make_t(glist, m_g_unitvec_1, m_g_unitvec_2, valley):
    """
    calculate interlayer interaction hamiltonian element
    """

    glist_size = np.shape(glist)[0]
    tmat = np.zeros((2*glist_size, 2*glist_size), complex)
    (g1, g2, g3, t1, t2, t3) =_make_transfer_const(m_g_unitvec_1, m_g_unitvec_2, valley)

    for i in range(glist_size):
        for j in range(glist_size):
            delta_k = glist[j] - glist[i]
            # matrix element in three cases:
            if _check_eq(delta_k, g1):
                tmat[2*i:2*i+2,2*j:2*j+2] = t1
            if _check_eq(delta_k, g2):
                tmat[2*i:2*i+2,2*j:2*j+2] = t2
            if _check_eq(delta_k, g3):
                tmat[2*i:2*i+2,2*j:2*j+2] = t3

    return tmat 


def _make_h1(glist, k, kpt1, rotmat, valley):
    """
    calculate first layer hamiltonian, approximated by dirac hamiltonian
    """

    glist_size = np.shape(glist)[0]
    h1mat = np.zeros((2*glist_size, 2*glist_size), complex)
    
    for i in range(glist_size):
        q = k + glist[i] - valley*kpt1
        dirac = -HBARVF*(valley*SIGMA_X*q[0]+SIGMA_Y*q[1])
        h1mat[2*i:2*i+2, 2*i:2*i+2] = dirac
        
    return h1mat


def _make_h2(glist, k, kpt2, rotmat, valley):
    """
    calculate second layer hamiltonian, approximated by dirac hamiltonian
    """

    glist_size = np.shape(glist)[0]
    h2mat = np.zeros((2*glist_size, 2*glist_size), complex)
    
    for i in range(glist_size):
        q = k + glist[i] - valley*kpt2
        dirac = -HBARVF*(valley*SIGMA_X*q[0]+ SIGMA_Y*q[1])
        h2mat[2*i:2*i+2, 2*i:2*i+2] = dirac
        
    return h2mat


def _make_hamk(k, kpt1, kpt2, m_g_unitvec_1, m_g_unitvec_2, glist, rt_mtrx_half, tmat, valley):
    """
    generate total hamiltonian 
    """

    h1mat  = _make_h1(glist, k, kpt1, rt_mtrx_half, valley)
    h2mat  = _make_h2(glist, k, kpt2, rt_mtrx_half.T, valley)
    hamk   = np.block([[h1mat,                      tmat], 
                      [np.conj(np.transpose(tmat)), h2mat]])
    
    return hamk


def _set_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, nk):
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
            kline[ikpt+i*nk] = kline[i*nk-1] + ikpt*step   
            kmesh[ikpt+i*nk] = vec*ikpt/(nk-1) + ksec[i]

    return (kline, kmesh)


def cont_solver(n_moire, n_g, n_k, valley):
    """
    continuum model solver for TBG system
    """

    (m_unitvec_1,   m_unitvec_2, m_g_unitvec_1, 
     m_g_unitvec_2, m_gamma_vec, m_k1_vec,    
     m_k2_vec,      m_m_vec,     rt_mtrx_half) = _set_moire(n_moire)

    kpt1, kpt2 = _set_kpt(rt_mtrx_half)
    glist  = _make_glist(n_g, n_moire, m_g_unitvec_1, m_g_unitvec_2, valley)
    tmat   = _make_t(glist, m_g_unitvec_1, m_g_unitvec_2, valley)
    kline, kmesh = _set_kmesh(m_gamma_vec, m_k1_vec, m_k2_vec, m_m_vec, n_k)

    emesh = []
    dmesh = []
    count = 1

    for k in kmesh:
        print("k sampling process, counter:", count)
        count += 1
        hamk = _make_hamk(k, kpt1, kpt2, m_g_unitvec_1, m_g_unitvec_2, 
                          glist,   rt_mtrx_half,  tmat, valley)
        eigen_val, eigen_vec = np.linalg.eigh(hamk)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)
    
    return (np.array(emesh), np.array(dmesh), kline)


if __name__ == "__main__":
    n_moire = 31
    n_g = 5
    n_k = 30
    valley = -1

    (emesh, dmesh, kline) = cont_solver(n_moire, n_g, n_k, valley)
    n_band = emesh[0].shape[0]

    fig, ax = plt.subplots()
    ax.set_xticks([kline[0], kline[n_k], kline[2*n_k-1], kline[3*n_k-1]])
    ax.set_xticklabels(["K1", "Γ", "M", "K2"])
    ax.set_xlim(0, kline[-1])

    # 7 bands
    for i in range(1):
        plt.plot(kline, emesh[:, n_band//2+i])
        plt.plot(kline, emesh[:, n_band//2-i])

    plt.plot(kline, emesh[:, n_band//2-1])
    #plt.plot(kline, emesh[:, n_band//2-2])

    ax.set_ylabel("Engergy (eV)")
    ax.set_title("Continuum Model, Band Structure of TBG, Nmoire: "+str(n_moire)+", Valley: "+str(valley))
    ax.axvline(x=kline[0], color="black")
    ax.axvline(x=kline[n_k-1], color="black")
    ax.axvline(x=kline[2*n_k-1], color="black")
    ax.axvline(x=kline[3*n_k-1], color="black")

    plt.savefig("../fig/continuum.png", dpi=500)
