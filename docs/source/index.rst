.. mtbmtbg documentation master file, created by
   sphinx-quickstart on Fri Jun 17 13:36:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tight Binding Planewave Method for Twisted Bilayer Graphene
===========================================================

Tight Binding Planewave Expansion (TBPLW) is a highly efficient and accurate numerical scheme to determine the low energy electronic
spectrum of Twisted Bilayer Graphene (TBG) inspired by continuum model. Our method utilizes real space information of carbon atoms in the super unit cell 
and projects the full Tight Binding Hamiltonian into a much smaller subspace using the atomic Bloch wavefunction basis.

Features
========

- Tight Binding Solver for TBG problem: `TBPLW`, `TBFULL`, `TBSPASE` methods implemented.
- Continumm model solver.
- Different TB method band structure comparison.
- Moire potential Analysis.
- Valley Chern number calculation.
- :math:`C_3` symmetry constrain.

Dependent Module
================
 
Our development is based on `SciPy`, `Numpy`, `Sklearn`, `Pybinding` and `matplotlib`.

RoadMap
=======

.. image:: ../../assets/roadmap.svg


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   benchmark
   api
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
