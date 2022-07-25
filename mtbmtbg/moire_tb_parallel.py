import time
import numpy as np
from pip import main
import scipy.linalg as sla
from scipy import sparse
from mpi4py import MPI
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
import moire_io as mio
from mtbmtbg.config import TBInfo, DataType, EngineType, ValleyType
import moire_tb as mtb
    
def tb_solver_parallel(n_moire: int,
              n_g: int,
              n_k: int,
              disp: bool = True,
              datatype=DataType.CORRU,
              engine=EngineType.TBPLW,
              valley=ValleyType.VALLEYK1) -> dict:
    """tight binding solver for TBG

    Args:
        n_moire (int): an integer describing the size of commensurate TBG systems
        n_g (int): Glist size, n_g = 5 for MATBG
        n_k (int): n_k 
        disp (bool): whether calculate dispersion
        datatype (DataType, optional): atom data type. Defaults to DataType.CORRU.
        engine (EngineType, optional): TB solver engine type. Defaults to EngineType.TBPLW.
        valley (EngineType, optional): valley concerned. Defaults to EngineType.VALLEYK1.

    Returns:
        dict:         
        'emesh': np.array(emesh),
        'dmesh': np.array(dmesh),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    """         

    # Allocation
    dmesh = []
    emesh = []
    kline = 0
    emax = -1000
    emin = 1000
    count = 1

    # Start of MPI
    start_time = time.process_time()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    p = comm.Get_size()


    # Initialization
    if rank == 0:
        # load atom data
        atom_pstn_list = mio.read_atom_pstn_list(n_moire, datatype)
        # construct moire info
        (_, m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
        (all_nns, enlarge_atom_pstn_list) = mset.set_atom_neighbour_list(atom_pstn_list, m_basis_vecs)
        (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atom_pstn_list, enlarge_atom_pstn_list, all_nns)
        # set up g list
        o_g_vec_list = mgk.set_g_vec_list(n_g, m_basis_vecs)
        # move to specific valley or combined valley
        g_vec_list = mtb._set_g_vec_list_valley(n_moire, o_g_vec_list, m_basis_vecs, valley)
        # constant matrix dictionary
        const_mtrx_dict = mtb._set_const_mtrx(n_moire, npair_dict, ndist_dict, m_basis_vecs, g_vec_list, atom_pstn_list)
        # constant list
        (transmat_list, neighbor_map) = mgk.set_kmesh_neighbour(n_g, m_basis_vecs, o_g_vec_list)

        if disp:
            (kline, kmesh) = mgk.set_tb_disp_kmesh(n_k, high_symm_pnts)
        else:
            kmesh = mgk.set_kmesh(n_k, m_basis_vecs)
        n_atom = atom_pstn_list.shape[0]
        n_band = g_vec_list.shape[0]*4
        n_kpts = kmesh.shape[0]
        print("="*100)
        print("num of atoms".ljust(30), ":", n_atom)
        print("num of kpoints".ljust(30), ":", n_kpts)
        print("num of bands".ljust(30), ":", n_band)
        print("="*100)
    

        atomdata = {'ndist_dict':ndist_dict, 'npair_dict':npair_dict, 'n_atom':n_atom, 'const_mtrx_dict':const_mtrx_dict}
        emesh_recv = np.empty((n_kpts, n_band),dtype='double')
        dmesh_recv = np.empty((n_kpts, n_band, n_band),dtype='double') # make sure we use len(), is it always 244?
        # Splits data and sends data to different cores
        alldata = np.array_split(kmesh, p)
        for n in range(1, p):
            data = alldata[n]
            numData = data.shape[0]
            comm.send(numData, dest=n)
            comm.Send(data, dest=n)
    else:
        atomdata = None
    atomdata = comm.bcast(atomdata, root=0)

    print('The current rank is: ', rank)
    if rank ==0:
        mesh1 = alldata[0]
    else:
        emesh_recv = None
        dmesh_recv = None
        numData = comm.recv(source=0)
        mesh1 = np.empty((numData,2))
        comm.Recv(mesh1, source=0)
            
        # assign atom data received from root
        ndist_dict = atomdata['ndist_dict']
        npair_dict = atomdata['npair_dict']
        const_mtrx_dict = atomdata['const_mtrx_dict']
        n_atom = atomdata['n_atom']
    setup_time = time.process_time()

    # Calculations
    for k_vec in mesh1:
        print("k sampling process, counter:", count)
        count += 1
        hamk = mtb._cal_hamiltonian_k(ndist_dict, npair_dict, const_mtrx_dict, k_vec, n_atom, engine)
        eigen_val, eigen_vec = mtb._cal_eigen_hamk(hamk, const_mtrx_dict['sr'], datatype, engine)
        if np.max(eigen_val)>emax:
            emax = np.max(eigen_val)
        if np.min(eigen_val)<emin:
            emin = np.min(eigen_val)
        emesh.append(eigen_val)
        dmesh.append(eigen_vec)

    emesh_sends = np.array(comm.gather(len(emesh)*len(emesh[0]), 0))
    dmesh_sends = np.array(comm.gather(len(dmesh)*2*len(dmesh[0])*len(dmesh[1]), 0))
    emesh_local = np.array(emesh)
    dmesh_local = np.array(dmesh)

    # recombnining emesh/dmesh array
    comm.Gatherv(sendbuf=emesh_local, recvbuf=(emesh_recv, emesh_sends), root = 0)
    comm.Gatherv(sendbuf=dmesh_local, recvbuf=(dmesh_recv, dmesh_sends), root = 0)
    if rank == 0:
        return {
        'emesh': np.array(emesh_recv),
        'dmesh': np.array(dmesh_recv),
        'kline': kline,
        'trans': transmat_list,
        'nbmap': neighbor_map
    }

    comp_time = time.process_time()

    print("="*100)
    print("emax =", emax, "emin =", emin)
    print("="*100)
    print("set up time:", setup_time-start_time, "comp time:", comp_time-setup_time)
    print("="*100)

    MPI.Finalize()