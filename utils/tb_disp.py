import sys
sys.path.append("..")
import tightbinding.moire_plot as mplt

n_moire = 30
n_g = 5
n_k = 20

mplt.tightbinding_plot_sep_valley(n_moire, n_g, n_k, 1, 'symm_relax', "./test_newsymm")
mplt.tightbinding_plot_valley_comb(n_moire, n_g, n_k, 2, 'symm_relax', "./test_newsymm")
