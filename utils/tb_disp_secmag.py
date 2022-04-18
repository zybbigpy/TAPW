import sys
sys.path.append("..")


import numpy as np
import tightbinding.moire_tb as tbtb


n_g = 6
n_k = 30
# Control band plotted
band = 5

for n_moire in range(55, 58):
    tbtb.tightbinding_plot(n_moire, n_g, n_k, band, True, relax=False)
    tbtb.tightbinding_plot(n_moire, n_g, n_k, band, True, relax=True)