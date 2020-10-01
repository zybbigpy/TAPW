import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import numpy as np

# should be two
print(np.dot(tbset.A_UNITVEC_1, tbset.A_G_UNITVEC_1)/np.pi)
print(np.dot(tbset.A_UNITVEC_2, tbset.A_G_UNITVEC_2)/np.pi)
tbset.system_info_log(30)
atoms = tbset.set_atom_list(16)
tbset.save_atom_list(atoms)
# check 