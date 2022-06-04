import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_symgen as msymm
import matplotlib.pyplot as plt


class MoireSymmGenTest(unittest.TestCase):

    def test_c3_group_gen(self):
        for n_moire in range(30, 65):
            nn_c31, nn_c32 = msymm.cal_c3_group(n_moire)
            # group = np.load("../data/group/group"+str(n_moire)+".npy")
            # self.assertTrue(np.allclose(nn_c31, group[:, 0]))
            # self.assertTrue(np.allclose(nn_c32, group[:, 1]))
