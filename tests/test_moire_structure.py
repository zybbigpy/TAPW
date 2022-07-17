import sys
import unittest

sys.path.append("..")

import numpy as np
import mtbmtbg.moire_setup as mset
import mtbmtbg.moire_io as mio
import matplotlib.pyplot as plt

from mtbmtbg.config import DataType



class MoireStructureTest(unittest.TestCase):



    def test_atom_corru_generation(self):

        for n_moire in range(25, 50):
            # genarate TBG with corrugation
            atoms_gen = mset.set_atom_pstn_list(n_moire)
            # atoms load
            atoms_load = mio.read_atom_pstn_list(n_moire)
            
            self.assertTrue(np.allclose(atoms_gen, atoms_load))

    def test_atom_rigid_generation(self):

        for n_moire in range(25, 50):
            # genarate TBG as a rigid structure
            atoms_gen = mset.set_atom_pstn_list(n_moire, corru=False)
            # atoms load
            atoms_load = mio.read_atom_pstn_list(n_moire, DataType.RIGID)

            self.assertTrue(np.allclose(atoms_gen, atoms_load))