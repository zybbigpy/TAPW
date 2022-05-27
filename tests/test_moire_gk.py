import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
import matplotlib.pyplot as plt


class MoireGKTest(unittest.TestCase):

    def test_glist_generation(self):
        ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(30)
        glist = mgk.set_g_vec_list(5, m_basis_vecs)
        self.assertTrue(np.allclose(glist[0], np.array([0.0, 0.0])))

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        for i in range(glist.shape[0]):
            ax.scatter(glist[i][0], glist[i][1], marker='o', s=400, c='w', edgecolors='k', alpha=0.4)
            ax.text(
                glist[i][0],
                glist[i][1],
                str(i),
                ha="center",
                va="center",
            )
        ax.set_title('Glist')
        plt.tight_layout()
        plt.savefig("./test_glist.png", dpi=500)

    def test_kmesh_genetation(self):
        ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(30)
        kmesh = mgk.set_kmesh(10, m_basis_vecs)
        self.assertTrue(np.allclose(kmesh[0], np.array([0.0, 0.0])))

        fig, ax = plt.subplots()
        for i in range(kmesh.shape[0]):
            ax.scatter(kmesh[i][0], kmesh[i][1], marker='o', s=400, c='w', edgecolors='k', alpha=0.4)
            ax.text(
                kmesh[i][0],
                kmesh[i][1],
                str(i),
                ha="center",
                va="center",
            )
        ax.set_title('kmesh')
        plt.tight_layout()
        plt.savefig("./test_kmesh.png", dpi=500)

    def test_kdisp_generation(self):
        ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(30)
        n_k1 = 15
        n_k2 = 20
        n_k3 = 30
        (kline1, kmesh1) = mgk.set_tb_disp_kmesh(n_k1, high_symm_pnts)
        (kline2, kmesh2) = mgk.set_tb_disp_kmesh(n_k2, high_symm_pnts)
        (kline3, kmesh3) = mgk.set_tb_disp_kmesh(n_k3, high_symm_pnts)

        self.assertEqual(kline1[n_k1], kline2[n_k2])
        self.assertEqual(kline1[n_k1], kline3[n_k3])
        self.assertEqual(kline1[2*n_k1], kline2[2*n_k2])
        self.assertEqual(kline1[2*n_k1], kline3[2*n_k3])
        self.assertEqual(kline1[3*n_k1], kline2[3*n_k2])
        self.assertEqual(kline1[3*n_k1], kline3[3*n_k3])
