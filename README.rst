THCPP
=====

**THCPP** is a C++ implementation of the tensor hypercontraction algorithm to factorize
the two electron integrals. In particular it implements the interpolative separable
density fitting algorithm with centroidal Veronoi tesselation algorithm [#]_.

The code will write factorized two-electron integrals in the `QMCPACK format
<https://qmcpack.readthedocs.io/en/develop/afqmc.html#listing-54>`_, for use in auxiliary
field quantum Monte Carlo simulations [#]_.

.. image:: http://img.shields.io/badge/License-LGPL%20v2.1-blue.svg
    :target: http://github.com/fdmalone/pauxy/blob/master/LICENSE

Requirements
------------

To build the code you will require:

- fftw3
- mpi
- scalapack
- hdf5
- cmake

Building
--------

To compile do (assuming mkl):

.. code-block:: bash

    mkdir build && cd build
    CC=mpicc CXX=mpic++ cmake -DMKL_FLAG=cluster -DMPIEXC_EXECUTABLE=srun -DMPIEXEC_NUMPROC_FLAG=-n ../

Testing
-------

Unit tests can be run with ctest:

.. code-block:: bash

    cd build
    ctest

References
----------

.. [#] Kun Dong, Wei Hu, and Lin Lin J. Chem. Theory Comput. 14, 1311 (2018)
.. [#] Fionn D Malone, Shuai Zhang, and Miguel A Morales, J. Chem. Theory Comput. 15, 256 (2019)
