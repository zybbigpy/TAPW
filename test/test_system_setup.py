import sys
sys.path.append("..")

import tightbinding.system_setup as tbset
import numpy as np

# should be two
print(np.dot(tbset.a_unitvec_1, tbset.a_g_unitvec_1)/np.pi)
print(np.dot(tbset.a_unitvec_2, tbset.a_g_unitvec_2)/np.pi)
print(tbset.set_moire(30))
# check 