import numpy as np
import scipy.linalg as la

import mtbmtbg.moire_tb as mtb
from mtbmtbg.config import DataType, ValleyType


def index(x, y, n_k):
    """determine the index in the kmesh

    Args:
        x (int): 
        y (int): 
        n_k (int): 

    Returns:
        int: 
    """
    return (x % n_k)*n_k+(y % n_k)


def d(x, y, n_k):
    """check whether the B.Z. boudary

    Args:
        x (_type_): _description_
        y (_type_): _description_
        n_k (_type_): _description_

    Returns:
        _type_: _description_
    """

    dx, dy = 0, 0

    if x == n_k:
        dx = 1
    if y == n_k:
        dy = 1

    return (dx, dy)


def braket_norm(phi1, phi2, x1, y1, x2, y2, n_k, trans, nmap):

    dx1, dy1 = d(x1, y1, n_k)
    dx2, dy2 = d(x2, y2, n_k)
    amat1 = trans[nmap[0, dx1, dy1]]
    amat2 = trans[nmap[0, dx2, dy2]]
    braket = ((amat1.T)@phi1).transpose().conj().dot(amat2.T@phi2)
    res_det = la.det(braket)

    return res_det/la.norm(res_det)


def ux(bands, x, y, n_k, init, last, transmat_list, neighbor_map):
    phi1 = bands[index(x, y, n_k)][:, init:last+1]
    phi2 = bands[index(x+1, y, n_k)][:, init:last+1]

    return braket_norm(phi1, phi2, x, y, x+1, y, n_k, transmat_list, neighbor_map)


def uy(bands, x, y, n_k, init, last, transmat_list, neighbor_map):
    phi1 = bands[index(x, y, n_k)][:, init:last+1]
    phi2 = bands[index(x, y+1, n_k)][:, init:last+1]

    return braket_norm(phi1, phi2, x, y, x, y+1, n_k, transmat_list, neighbor_map)


def small_loop(bands, m, n, n_k, init, last, transmat_list, neighbor_map):

    return np.log(
        ux(bands, m, n, n_k, init, last, transmat_list, neighbor_map)*
        uy(bands, m+1, n, n_k, init, last, transmat_list, neighbor_map)/
        ux(bands, m, n+1, n_k, init, last, transmat_list, neighbor_map)/
        uy(bands, m, n, n_k, init, last, transmat_list, neighbor_map))


def cal_chern(bands, n_k, init, last, transmat_list, neighbor_map):

    ret = 0

    for m in range(n_k):
        for n in range(n_k):
            ret += small_loop(bands, m, n, n_k, init, last, transmat_list, neighbor_map)

    return ret/(2*np.pi*1j)


def cal_moire_chern(n_moire: int, n_g: int, n_k: int, n_chern: int, datatype=DataType.CORRU,
                    valley=ValleyType.VALLEYK1):

    cherns = []
    ret = mtb.tb_solver(n_moire, n_g, n_k, disp=False, datatype=datatype, valley=valley)
    dmesh = ret['dmesh']
    trans = ret['trans']
    nmap = ret['nbmap']
    nband = dmesh.shape[2]
    dmesh = dmesh[:, :, (nband//2-n_chern):(nband//2+n_chern)]
    for i in range(2*n_chern):
        chern = cal_chern(dmesh, n_k, i, i, trans, nmap)
        assert np.imag(chern)<1e-9
        cherns.append(np.rint(np.real(chern)))
        print("band i:", i, "chern number:", np.rint(np.real(chern)))

    return np.array(cherns)
