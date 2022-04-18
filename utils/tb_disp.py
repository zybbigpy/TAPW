import sys
sys.path.append("..")


import numpy as np
import tightbinding.moire_tb as tbtb


n_moire = 30
n_g = 5
n_k = 10
# Control band plotted
band = 1

tbtb.tightbinding_plot(n_moire, n_g, n_k, band, True, "indextTest", False)

