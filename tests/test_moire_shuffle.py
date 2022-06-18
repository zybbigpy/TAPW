import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_shuffle as mshuffle
import mtbmtbg.moire_cont as mcont
import mtbmtbg.moire_plot as mplot

import matplotlib.pyplot as plt


class MoireShuffleTest(unittest.TestCase):

    def testContShuffle2TBPLW(self):
        ret = mcont.cont_potential(30, 5)
        glist = ret['glist']
        u_val = ret['mpot']
        fig, ax = plt.subplots()
        mplot.glist_plot_module(ax, glist, u_val)
        plt.tight_layout()
        plt.savefig("./shuffle_potential.png", dpi=500)
        plt.close()