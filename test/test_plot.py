import sys
sys.path.append("..")
import tightbinding.moire_plot as mplt


n_moire = 13
n_g = 5
n_k = 20


mplt.tightbinding_plot_sparsetb(n_moire, n_g, n_k, 5, 'atom', "./test")
mplt.tightbinding_plot_fulltb(n_moire, n_g, n_k, 10, 'atom', "./test")
mplt.tightbinding_plot_fulltb(n_moire, n_g, n_k, 10, 'corrugation', "./test")



n_moire = 30
n_g = 5
n_k = 20

mplt.tightbinding_plot_sep_valley(n_moire, n_g, n_k,  1, 'atom', "./test")
mplt.tightbinding_plot_valley_comb(n_moire, n_g, n_k, 2, 'atom', "./test")
mplt.tightbinding_plot_sep_valley(n_moire, n_g, n_k,  1, 'corrugation', "./test")
mplt.tightbinding_plot_valley_comb(n_moire, n_g, n_k, 2, 'corrugation', "./test")
mplt.tightbinding_plot_sep_valley(n_moire, n_g, n_k,  1, 'relax', "./test", "test", mu=True)
mplt.tightbinding_plot_valley_comb(n_moire, n_g, n_k, 2, 'relax', "./test")
mplt.tightbinding_plot_sep_valley(n_moire, n_g, n_k,  1, 'symm_relax', "./test")
mplt.tightbinding_plot_valley_comb(n_moire, n_g, n_k, 2, 'symm_relax', "./test")
