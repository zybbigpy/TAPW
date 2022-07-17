import numpy as np
from mtbmtbg.config import DataType


def read_atom_pstn_list(n_moire: int, datatype=DataType.CORRU) -> np.ndarray:
    """read atom position list from csc files

    Args:
        n_moire (int): an integer to describe a moire TBG
        datatype (DataType, optional): the datatype for atoms. Defaults to DataType.CORRU.

    Returns:
        np.ndarray: atom_pstn_list array
    """

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
            "../data/rigid/atom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    elif datatype == DataType.CORRU:
        print("Load corrugation data.")
        atom_pstn_list = np.loadtxt(
            "../data/corrugation/atom"+str(n_moire)+".csv",
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
