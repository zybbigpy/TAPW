import numpy as np
from mtbmtbg.config import DataType


def read_atom_pstn_list(n_moire: int, datatype=DataType.CORRU) -> np.ndarray:

    if datatype == DataType.RELAX:
        print("Load relaxed data after symmetrized.")
        atom_pstn_list = np.loadtxt(
            "../data/relaxsymm/symmatom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    elif datatype == DataType.RIGID:
        print("Load rigid atomic data.")
        atom_pstn_list = np.loadtxt(
            "../data/atom/atom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    elif datatype == DataType.CORRU:
        print("Load corrugation data.")
        atom_pstn_list = np.loadtxt(
            "../data/atom/atom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    else:  # default datatype == DataType.CORRU
        print("Default! Load corrugation data.")
        atom_pstn_list = np.loadtxt(
            "../data/corrugation/atom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )

    return atom_pstn_list
