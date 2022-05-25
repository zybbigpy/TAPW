import numpy as np


def read_atom_pstn_list(n_moire: int, datatype: str) -> list:

    if datatype == "symm_relax":
        print("Load relaxed data after symmetrized.")
        atom_pstn_list = np.loadtxt(
            "../data/relaxsymm/symmatom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    elif datatype == "relax":
        print("Load relaxed data.")
        atom_pstn_list = np.loadtxt(
            "../data/relax/relaxatom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    elif datatype == "corrugation":
        print("Load corruagtion data.")
        atom_pstn_list = np.loadtxt(
            "../data/corrugation/atom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )
    elif datatype == "atom":
        print("Load atomic data.")
        atom_pstn_list = np.loadtxt("../data/atom/atom"+str(n_moire)+".csv", delimiter=",", comments="#")
    else:
        print("Default! Load corrugation data.")
        atom_pstn_list = np.loadtxt(
            "../data/corrugation/atom"+str(n_moire)+".csv",
            delimiter=",",
            comments="#",
        )

    return list(atom_pstn_list)


def save_atom_pstn_list(atom_pstn_list: list, path: str, n_moire: int):

    atoms = np.array(atom_pstn_list)
    np.savetxt(path+"atom"+str(n_moire)+".csv", atoms, header="Rx, Ry, d", delimiter=",")


def read_atom_neighbour_list(path: str, n_moire: int) -> list:
    """
    aborted, we wont generate neighbour list file any more.
    """

    with open(path+"Nlist"+str(n_moire)+".dat", "r") as f:
        print("Open file Nlist...\n")
        lines = f.readlines()
        atom_neighbour_list = []
        for line in lines:
            line_data = line.split()
            data = [int(data_str) for data_str in line_data]
            atom_neighbour_list.append(data)

    return atom_neighbour_list
