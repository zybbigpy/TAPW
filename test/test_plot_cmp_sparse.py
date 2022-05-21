import sys

sys.path.append("..")
import tightbinding.moire_plot as mplt

n_moire = 10
n_g = 5
n_k = 30

bandfull = 5
bandpw = 10
mplt.sparsetb_comb_valley_cmp(n_moire, n_g, n_k, bandfull, bandpw, "atom", "./test")
bandfull = 5
bandpw = 5
mplt.sparsetb_sep_valley_cmp(n_moire, n_g, n_k, bandfull, bandpw, "atom", "./test")
