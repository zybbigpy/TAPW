import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import magnetic.moire_magnetic_tb as mgtb


VALLEY_1 = 1
VALLEY_2 = -1

n_moire = 30
n_g = 5
n_k = 8

# pq_list = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4), (1, 5),
#            (2, 5), (3, 5), (4, 5), (1, 6), (5, 6), (1, 7),
#            (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (1, 8),
#            (3, 8), (5, 8), (7, 8), (1, 9), (2, 9), (4, 9),
#            (5, 9), (7, 9), (8, 9),(1, 10),(3, 10),(7, 10),
#           (9, 10),(1, 11),(2, 11),(3, 11),(4, 11),(5, 11),
#           (6, 11),(7, 11),(8, 11),(9, 11),(10,11),(1, 12),
#           (5, 12),(7, 12),(11,12),(1, 13),(2, 13),(3, 13),
#           (4, 13),(4, 13),(5, 13),(6, 13),(7, 13),(8, 13),
#           (9, 13),(10,13),(11,13),(12,13),(1, 14),(3, 14),
#           (5, 14),(9, 14),(11,14),(13,14),(1, 15),(2, 15),
#           (4, 15),(7, 15),(8, 15),(11,15),(13,15),(14,15),
#           (1, 16),(3, 16),(5, 16),(7, 16),(9, 16),(11,16)]

pq_list = [(p+1, 30) for p in range(29)]


e_total = []

for p, q in pq_list:
    (eig, eigfun) = mgtb.mag_tb_solver_periodic(n_moire=16, n_g=5, n_k=1, valley=VALLEY_1, p=p, q=q, type=1, disp=False)
    e_total.append(eig)

np.save("./etotal_v_+1.npy", e_total)   


# for p, q in pq_list:
#     (emesh, dmesh) = mgtb.mag_tb_solver(n_moire, n_g, n_k, VALLEY_1, p, q, disp=False)
#     emesh = np.array(emesh)
#     e_total.append(emesh)
#     # n_band = emesh[0].shape[0]
#     # print(emesh.shape)
#     # emesh_reshaped = (emesh[:, n_band//2-2:n_band//2]).reshape(-1)
#     # print(emesh_reshaped.shape)
#     # pq = np.repeat(p/q, emesh_reshaped.shape[0])
#     # print(pq.shape)
#     # plt.plot(pq, emesh_reshaped,'ko', markersize=0.2)

#     # plt.savefig("./test3.png")``
# np.save("./etotal_v_+1.npy", e_total)


# etotal =  np.load("./etotal_v_-1.npy")
# print(etotal.shape)

# fig, ax = plt.subplots()
# #ax.set_ylim(0.825, 0.8267)
# ax.set_ylim(0.725, 0.9267)

# #ax.set_ylim(0.7, 0.8)


# for i in range(60):
#     p,q = pq_list[i]
#     emesh = etotal[i]
#     print(emesh.shape)
#     n_band = emesh[0].shape[0]
#     emesh_reshaped= (emesh[:, n_band//2-3:n_band//2+3]).reshape(-1)
#     #emesh_reshaped= emesh.reshape(-1)
#     pq = np.repeat(p/q, emesh_reshaped.shape[0])
#     plt.plot(pq, emesh_reshaped,'ko', markersize=0.3)

# etotal1 =  np.load("./etotal_v_+1.npy")
# for i in range(60):
#     p,q = pq_list[i]
#     emesh1 = etotal1[i]
#     print(emesh1.shape)
#     n_band1 = emesh1[0].shape[0]
#     emesh_reshaped1= (emesh1[:, n_band1//2-3:n_band1//2+3]).reshape(-1)
#     #emesh_reshaped= emesh.reshape(-1)
#     pq = np.repeat(p/q, emesh_reshaped1.shape[0])
#     plt.plot(pq, emesh_reshaped1,'ko', markersize=0.3)

# plt.savefig("./test4.png")