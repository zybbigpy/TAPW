import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_chern as mchern


class MoireChernTest(unittest.TestCase):

    # def test_cal_chern(self):
    #     dmesh = np.load("../tests_files/bands.npy")
    #     trans = np.load("../tests_files/trans.npy")
    #     nmap = np.load("../tests_files/map.npy")
    #     n_k = 10
    #     print(dmesh.shape, trans.shape, nmap.shape)
    #     nband = dmesh.shape[2]
    #     nchern = 5
    #     dmesh = dmesh[:, :, (nband // 2-nchern):(nband // 2+nchern)]
    #     for i in range(2*nchern):
    #         chern = mchern.cal_chern(dmesh, n_k, i, i, trans, nmap)
    #         assert np.imag(chern)<1e-9
    #         print("band i:", i, "chern number:", np.rint(np.real(chern)))

    def test_chern_calculation(self):
        n_moire = 30
        n_k = 10
        n_g = 5
        n_chern = 5
        cherns = mchern.cal_moire_chern(n_moire, n_g, n_k, n_chern)
        self.assertTrue(np.allclose(cherns, np.array([-5, 1, 0, 1, 1, -1, -1, 0, 0, -2])))