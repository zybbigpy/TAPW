import numpy as np
import matplotlib.pyplot as plt

for nmoire in range(30, 45, 2):
    name = "atom"+str(nmoire)
    atom_pstn_list = np.loadtxt("../data/"+name+".csv", delimiter=',', comments='#')

    print(atom_pstn_list.shape)
    #plt.plot(atom_pstn_list[:,0], atom_pstn_list[:, 1])
    #plt.show()
    atom_dev_list = np.loadtxt(name+"-dr.dat", comments='#')
    print(atom_dev_list.shape)
    atom_dev = atom_dev_list[:, 1:]
    print(atom_dev.shape)

    assert atom_dev.shape == atom_pstn_list.shape

    newatom_pstn_list = atom_pstn_list+atom_dev
    print(newatom_pstn_list.shape)
    np.savetxt("relax"+name+".csv", newatom_pstn_list, header="Rx, Ry, d", delimiter=',')
