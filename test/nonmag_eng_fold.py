import sys
sys.path.append("..")


import numpy as np
import matplotlib.pyplot as plt
import magnetic.moire_magnetic_tb as magtb


eng_fold = np.load("../data/emesh_n30_p0_q2_v-1.npy")
eng_mag  = np.load("../data/emesh_n30_p1_q2_v-1.npy")
print(eng_fold.shape)
n_band = eng_fold.shape[1]
ind1 = n_band//2-5
ind2 = n_band//2+5
print(eng_fold[0, ind1:ind2])
print(eng_mag[1, ind1:ind2])
print((eng_mag[1, ind1:ind2]-eng_fold[0, ind1:ind2])*100)