import numpy as np
import scipy.linalg as la

from scipy.linalg import block_diag
from scipy import sparse


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