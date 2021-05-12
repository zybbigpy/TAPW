import sys
sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
import magnetic.moire_magnetic_tb as magtb


VALLEY_2 = -1
n_moire = 30
n_g = 5
n_k = 10
p   = 1
q   = 6

emesh, emesh_proj = magtb.mag_tb_project(n_moire, n_g, n_k, VALLEY_2, p, q)

for x in range(n_k):
    for bnd in range(2*q):
        plt.scatter(x, emesh[x, bnd], s=1, c='red')
        plt.scatter(x, emesh_proj[x, bnd], s=1, c='blue')

plt.xlabel("k point")
plt.ylabel("Energy (eV)")
plt.title("red points for eigen value, blue point for eigen value project,valley:-1")
plt.savefig(str(q)+".png",dpi=500)