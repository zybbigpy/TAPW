import sys
sys.path.append("..")

import numpy as np
import tightbinding.moire_tb as tbtb
import tightbinding.moire_setup as tbset
import matplotlib.pyplot as plt

n_g = 7
n_k = 10

flatnessv1 = []
flatnessv2 = []


def cal_flatness(emesh):
    nband = emesh[0].shape[0]
    e1 = emesh[:, nband//2]
    e2 = emesh[:, nband//2-1]
    e = []
    print(e1.shape, e2.shape)
    for i in range(e1.shape[0]):
        e.append(e1[i])
    for i in range(e2.shape[0]):
        e.append(e2[i])

    return np.var(np.array(e))


# for n_moire in range(30, 45, 2):

#     emeshvp1, _, _, _, _ = tbtb.tightbinding_solver(n_moire, n_g, n_k,  1, False, True, True)
#     print(emeshvp1.shape)
#     flat = cal_flatness(emeshvp1)
#     flatnessv1.append(np.array([n_moire, flat]))

#     emeshvn1, _, _, _, _ = tbtb.tightbinding_solver(n_moire, n_g, n_k, -1, False, True, True)
#     print(emeshvn1.shape)
#     flat = cal_flatness(emeshvn1)
#     flatnessv2.append(np.array([n_moire, flat]))

# print(flatnessv1, flatnessv2)
# np.save("flatnessv1_30.npy", np.array(flatnessv1))
# np.save("flatnessv2_30.npy", np.array(flatnessv2))
# n_moire = [n for n in range(45, 70, 2)]
flatv1 = np.load("v1.npy")
flatv2 = np.load("v2.npy")

print(flatv1.shape, flatv1)

moire_angle = tbset.set_moire_angle
v1angle = []
v2angle = []

nmoirev1 = flatv1[:, 0]
nmoirev2 = flatv2[:, 0]

for n in nmoirev1:
    v1angle.append(moire_angle(int(n)))

for n in nmoirev2:
    v2angle.append(moire_angle(int(n)))

print(v1angle)
plt.plot(v1angle, flatv1[:, 1], 'o-', label='Valley 1')
plt.plot(v2angle, flatv2[:, 1], '*-', label='Valley 2')

plt.ylabel('flatness')
plt.xlabel('$\\theta$ (degree)')

plt.legend()
plt.savefig("./flatness.png", dpi=500)
