import sys
sys.path.append("..")

import tightbinding.moire_setup as tbset
import tightbinding.moire_tb as tbtb

import numpy as np

__author__ = 'Wangqian Miao'

n_moire = [i for i in range(16, 41)]

# print("="*100)
# test innder product between unit vec and reciprocal vec
assert(np.dot(tbset.A_UNITVEC_1, tbset.A_G_UNITVEC_1)/np.pi-2<1E-10)
assert(np.dot(tbset.A_UNITVEC_2, tbset.A_G_UNITVEC_2)/np.pi-2<1E-10)

# test info log
for n in n_moire:
    print("="*100)
    tbset.system_info_log(n)
    # test atom pstn list construction
    atoms = tbset.set_atom_pstn_list(n)
    tbset.save_atom_pstn_list(atoms, "../data/", n)