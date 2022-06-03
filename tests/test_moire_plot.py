import sys
import unittest

sys.path.append("..")

from mtbmtbg.config import DataType, EngineType, ValleyType
import mtbmtbg.moire_plot as mplot


class MoirePlotTest(unittest.TestCase):

    def test_plot_n10(self):
        n_moire = 10
        n_g = 5
        n_k = 20
        mplot.tb_plot_fulltb(n_moire, n_g, n_k, 5, datatype=DataType.RIGID)
        mplot.tb_plot_sparsetb(n_moire, n_g, n_k, 5, datatype=DataType.RIGID)

    def test_plot_n30(self):
        n_moire = 30
        n_g = 5
        n_k = 20
        mplot.tb_plot_tbplw_sepv(n_moire, n_g, n_k, 1, datatype=DataType.RIGID)
        mplot.tb_plot_tbplw_sepv(n_moire, n_g, n_k, 1, datatype=DataType.CORRU)
        mplot.tb_plot_tbplw_sepv(n_moire, n_g, n_k, 1, datatype=DataType.RELAX, figname='test')

        mplot.tb_plot_tbplw_combv(n_moire, n_g, n_k, 2, datatype=DataType.RIGID)
        mplot.tb_plot_tbplw_combv(n_moire, n_g, n_k, 2, datatype=DataType.CORRU)
        mplot.tb_plot_tbplw_combv(n_moire, n_g, n_k, 2, datatype=DataType.RELAX, mu=True)

    def test_fulltb_cmp(self):
        n_moire = 10
        n_g = 5
        n_k = 20
        mplot.fulltb_combv_cmp(n_moire, n_g, n_k, n_k, 10, 5, datatype=DataType.RIGID)
        mplot.fulltb_sepv_cmp(n_moire, n_g, n_k, n_k, 5, 5, datatype=DataType.RIGID)

    def test_sparsetb_cmp(self):
        n_moire = 10
        n_g = 5
        n_k = 20
        mplot.sparsetb_combv_cmp(n_moire, n_g, n_k, n_k-5, 10, 5, datatype=DataType.RIGID)
        mplot.sparsetb_sepv_cmp(n_moire, n_g, n_k, n_k-5, 5, 5, datatype=DataType.RIGID)
