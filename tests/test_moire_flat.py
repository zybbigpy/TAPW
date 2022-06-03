import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_flat as mflat


class MoireFlatTest(unittest.TestCase):

    def test_moire_flat(self):
        n_moire = 30
        n_g = 5
        v1, v2 = mflat.cal_flatness(n_moire, n_g)

        self.assertTrue(v1<2*1e-6)
        self.assertTrue(v2<2*1e-6)
