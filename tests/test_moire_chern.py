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


# test_dmesh = np.load("testd.np.npy")
# test_trans = np.load("testt.np.npy")
# test_nmap  = np.load("testn.np.npy")
# test_kmesh = np.load("test_kmesh.npy")
# test_gr_mtrx = np.load("test_gr_mtrx.npy")
# test_tr_mtrx = np.load("test_tr_mtrx_new.npy")
# test_dd = np.load("test_dd.npy")
# test_dr = np.load("test_dr.npy")
# test_hopping = np.load("test_hopping.npy")

# dmesh = np.load("../tests_files/bands.npy")
# trans = np.load("../tests_files/trans.npy")
# nmap  = np.load("../tests_files/map.npy")
# kmesh = np.load("../tests_files/kmesh.npy")
# gr_mtrx = np.load("../tests_files/gr_mtrx.npy")
# tr_mtrx = np.load("../tests_files/tr_mtrx_new.npy")
# dd = np.load("../tests_files/dd.npy")
# dr = np.load("../tests_files/dr.npy")
# hopping = np.load("../tests_files/hopping.npy")

# print(test_dmesh.shape, dmesh.shape)
# print(test_trans.shape, trans.shape)
# print(test_nmap.shape, nmap.shape)
# print(test_kmesh.shape, kmesh.shape)
# print(np.allclose(test_kmesh, kmesh))
# print(test_gr_mtrx.shape, gr_mtrx.shape)
# print(np.allclose(test_gr_mtrx, gr_mtrx))
# print(test_tr_mtrx.shape, tr_mtrx.shape)
# print(np.allclose(test_tr_mtrx, tr_mtrx))
# print(np.allclose(test_dr, dr))
# print(np.allclose(test_dd, dd))
# print(np.allclose(test_hopping, hopping))

# print(type(test_tr_mtrx), type(tr_mtrx))
# print(test_tr_mtrx[0][0], tr_mtrx[0][0])

# for i in range(61):
#     for j in range(2):
#         for k in range(2):
#             assert test_nmap[i][j][k]==nmap[i][j][k]

# for i in range(61):
#     for j in range(244):
#         for k in range(244):
#             assert test_trans[i][j][k]==trans[i][j][k]

# print(np.allclose(dmesh[0], test_dmesh[0]))
# print(dmesh[0][1][1], test_dmesh[0][1][1])
