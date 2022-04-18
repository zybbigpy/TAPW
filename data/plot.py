import numpy as np
import matplotlib.pyplot as plt

atom_pstn_list = np.loadtxt("atom65.csv", delimiter=',', comments='#')

print(atom_pstn_list.shape)
plt.plot(atom_pstn_list[:,0], atom_pstn_list[:, 1])
plt.show()