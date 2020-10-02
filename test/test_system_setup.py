import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import numpy as np

# test innder product between unit vec and reciprotocal vec
print(np.dot(tbset.A_UNITVEC_1, tbset.A_G_UNITVEC_1)/np.pi)
print(np.dot(tbset.A_UNITVEC_2, tbset.A_G_UNITVEC_2)/np.pi)

# test infor log
tbset.system_info_log(30)

# test atom pstn list construction
atoms = tbset.set_atom_pstn_list(16)
tbset.save_atom_pstn_list(atoms,"../data/", 16)

# test load neighour list, atom position list
atom_neigbour_list = tbset.read_atom_neighbour_list("../data/", 16)
atom_pstn_list = tbset.read_atom_pstn_list("../data/", 16)

print(atom_neigbour_list[2567][7])
print(atom_pstn_list[267] == atoms[267])
print(atom_pstn_list[1249] == atoms[1249])
