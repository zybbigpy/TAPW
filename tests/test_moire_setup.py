import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_setup as mset
import matplotlib.pyplot as plt


def read_atom_neighbour_list(path: str, n_moire: int) -> np.ndarray:
    with open(path+"Nlist"+str(n_moire)+".dat", "r") as f:
        lines = f.readlines()
        atom_neighbour_list = []
        for line in lines:
            line_data = line.split()
            data = [int(data_str) for data_str in line_data]
            atom_neighbour_list.append(data)

    return np.array(atom_neighbour_list)


class SetUpTest(unittest.TestCase):

    def test_atomic_structure(self):
        a1 = mset.A_UNITVEC_1
        a2 = mset.A_UNITVEC_2
        g1 = mset.A_G_UNITVEC_1
        g2 = mset.A_G_UNITVEC_2

        self.assertTrue(np.allclose(np.dot(a1, g1), 2*np.pi))
        self.assertTrue(np.allclose(np.dot(a2, g2), 2*np.pi))
        self.assertTrue(np.allclose(np.dot(a1, g2), 0))
        self.assertTrue(np.allclose(np.dot(a2, g1), 0))

    def test_moire_set(self):
        ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(30)
        mg1 = m_basis_vecs['mg1']
        mg2 = m_basis_vecs['mg2']
        mu1 = m_basis_vecs['mu1']
        mu2 = m_basis_vecs['mu2']
        self.assertTrue(np.allclose(np.dot(mu1, mg1), 2*np.pi))
        self.assertTrue(np.allclose(np.dot(mu2, mg2), 2*np.pi))
        self.assertTrue(np.allclose(np.dot(mu1, mg2), 0))
        self.assertTrue(np.allclose(np.dot(mu2, mg1), 0))

    def test_atom_generation(self):
        atoms = mset.set_atom_pstn_list(30)
        self.assertEqual(atoms.shape[0], 11164)

    def test_nn_generation(self):
        n_moire = 30
        atom_neighbour_list = read_atom_neighbour_list("../tests_files/", n_moire)
        atoms = mset.set_atom_pstn_list(n_moire)
        num_atoms = atoms.shape[0]
        ((rt_angle_r, rt_angle_d), m_basis_vecs, high_symm_pnts) = mset._set_moire(n_moire)
        all_nns, enlarge_atom_pstn_list = mset.set_atom_neighbour_list(atoms, m_basis_vecs)

        all_nns_mod = all_nns % num_atoms
        for i in range(num_atoms):
            arr_a = np.sort(all_nns_mod[i])
            arr_b = np.array(atom_neighbour_list[i])

            self.assertEqual(arr_a.shape, arr_b.shape)
            self.assertTrue(np.array_equal(arr_a, arr_b))

        (npair_dict, ndist_dict) = mset.set_relative_dis_ndarray(atoms, enlarge_atom_pstn_list, all_nns)
        self.assertEqual(ndist_dict['dd'].shape[0], ndist_dict['dr'].shape[0])
        self.assertEqual(len(npair_dict['c']), len(npair_dict['r']))