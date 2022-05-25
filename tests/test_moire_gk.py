import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_gk as mgk
import matplotlib.pyplot as plt


class SetUpTest(unittest.TestCase):

    def test_glist_generation(self):
        ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(30)
        glist = mgk._set_g_vec_list(5, m_basis_vecs)
        self.assertTrue(np.allclose(glist[0], np.array([0.0, 0.0])))

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        for i in range(glist.shape[0]):
            ax.scatter(glist[i][0], glist[i][1], marker='o', s=400, c='', edgecolors='black', alpha=0.4)
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
