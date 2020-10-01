import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import numpy as np

# should be two
print(np.dot(tbset.a_unitvec_1, tbset.a_g_unitvec_1)/np.pi)
print(np.dot(tbset.a_unitvec_2, tbset.a_g_unitvec_2)/np.pi)
tbset.system_info_log(30)
print(len(tbset.set_atom_list(16)))
# check 