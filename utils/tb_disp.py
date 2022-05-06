import sys
sys.path.append("..")


import numpy as np
import tightbinding.moire_tb as tbtb


n_moire = 30
n_g = 5
n_k = 20
# Control band plotted
band = 1

tbtb.tightbinding_plot(n_moire, n_g, n_k, band, True, "_4bnd", False)

#tbtb.tightbinding_plot_comb_valley(n_moire, n_g, n_k, band, True, "_4bnd", False)