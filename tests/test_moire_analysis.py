import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_analysis as manal
import mtbmtbg.moire_plot as mplot
from mtbmtbg.config import DataType, ValleyType, Structure

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch

class MoireAnaysisTest(unittest.TestCase):

    def test_moire_potential_analysis(self):
        n_moire = 30
        n_g = 5
        ret = manal.analyze_moire_potential(n_moire, n_g)
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

    def test_moire_potential_analysis_plot(self):
        n_moire = 30
        n_g = 5
        mplot.moire_potential_plot(n_moire, n_g, pathname="./test")
        mplot.moire_potential_plot(n_moire, n_g, kpnt='k1', datatype=DataType.RIGID, pathname="./test")

    def test_moire_band_convergence(self):
        n_moire = 30
        n_g = 5
        ret = manal.analyze_band_convergence(n_moire, n_g, datatype=DataType.CORRU, valley=ValleyType.VALLEYK1)
        glist = ret['glist']
        band = ret['band']['gamma']
        self.assertTrue(glist.shape[0], band.shape[0])

    def test_moire_band_convergence_plot(self):
        n_moire = 30
        n_g = 5
        mplot.moire_band_convergence_plot(n_moire, n_g, kpnt='m')
        mplot.moire_band_convergence_plot(n_moire, n_g, kpnt='k1')
        mplot.moire_band_convergence_plot(n_moire, n_g, kpnt='k2')
        mplot.moire_band_convergence_plot(n_moire, n_g, kpnt='gamma')

    
    def test_moire_vs_k(self):
        n_moire = 30
        n_g = 8
        n_k = 3
        ret = manal.moire_potential_vs_k(n_moire, n_g, n_k, datatype=DataType.CORRU)

        # [nm^-1] [meV]
        kpnts = ret['distance']*10
        maa = ret['moire_aa']*1000
        mab = ret['moire_ab']*1000

    
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        kx = kpnts[:, 0]
        ky = kpnts[:, 1]
        pat = Circle((Structure.ATOM_K_1[0]*10, Structure.ATOM_K_1[1]*10),
                    3.15,
                    edgecolor='k',
                    facecolor='white',
                    alpha=0.3,
                    linestyle=':')
        ax.add_patch(pat)
        p = ax.scatter(kx, ky, c=mab, marker='o', s=50, cmap='RdBu_r')
        cbar = fig.colorbar(p, ax=ax)
        cbar.ax.set_ylabel('$u\'$ [meV]', rotation=270, labelpad=12)
        mplot.adjust_spines(ax, ['left', 'bottom'])
        ax.set_xlabel('[nm$^{-1}$]')
        ax.set_ylabel('[nm$^{-1}$]')
        plt.tight_layout()
        plt.savefig("moire_pont_corru_ab_k.png", dpi=500)
        plt.close()
