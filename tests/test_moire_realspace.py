import sys
import unittest

sys.path.append("..")

from mtbmtbg.config import DataType, EngineType, ValleyType
import mtbmtbg.moire_plot as mplot


class MoireRealTest(unittest.TestCase):

    def test_moire_real_plot(self):
        n_moire = 10
        mplot.real_space_plot(n_moire)