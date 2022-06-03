import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_analysis as maly
import mtbmtbg.moire_plot as mplot
from mtbmtbg.config import DataType
import matplotlib.pyplot as plt


class MoireAnaysisTest(unittest.TestCase):

    def test_moire_analysis(self):
        n_moire = 30
        n_g = 5
        ret = maly.cal_moire_potential(n_moire, n_g)
        glist = ret['glist']
        self.assertTrue(np.allclose(glist[0], np.array([0, 0])))
        mpot = ret['mpot']
        u1_max = np.max(mpot['gamma']['u1'])
        u2_max = np.max(mpot['gamma']['u2'])
        u3_max = np.max(mpot['gamma']['u3'])
        u4_max = np.max(mpot['gamma']['u4'])
        # value by continuum model
        self.assertTrue(0.07<u1_max and u1_max<0.11)
        self.assertTrue(0.07<u2_max and u2_max<0.11)
        self.assertTrue(0.07<u3_max and u3_max<0.11)
        self.assertTrue(0.07<u4_max and u4_max<0.11)

    def test_moire_analysis_plot(self):
        n_moire = 30
        n_g = 5
        mplot.moire_potential_plot(n_moire, n_g, pathname="./test")
        mplot.moire_potential_plot(n_moire, n_g, kpnt='k1', datatype=DataType.RIGID, pathname="./test")
