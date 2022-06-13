import sys
import unittest

sys.path.append("..")

from mtbmtbg.config import DataType, EngineType, ValleyType
import mtbmtbg.moire_plot as mplot
import mtbmtbg.moire_cont as mcont


class MoireContTest(unittest.TestCase):

    def test_moire_bands(self):
        n_moire = 30
        n_g = 5
        n_k = 30
        mplot.cont_plot_combv(n_moire, n_g, n_k, 5)

    def test_flat_bands(self):
        n_moire = 31
        n_g = 5
        n_k = 30
        mplot.cont_plot_combv(n_moire, n_g, n_k, 1)
