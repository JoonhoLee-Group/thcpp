01 Carbon Diamond 2x2x2 pyscf
=============================


1. Run pyscf scf calculation

.. code-block:: bash

    python scf.py > scf.out

2. Dump (supercell) orbitals to file in format thcpp will read:

.. code-block:: bash

    python thcpp/tools/generate_thc_input.py -i scf.chk -o orbitals.h5

This will produce a file called orbitals.h5.

3. Run thcpp:

.. code-block:: bash

    srun -n 36 build/bin/thcpp.x input.json

Here we used 36 MPI tasks, use more / less as appropriate.

The input file is fairly self explanatory:

.. code-block:: json

    {
        "orbital_file": "orbitals.h5",
        "output_file": "thc.h5",
        "thc_cfac": 15,
        "thc_half_cfac": 10,
        "half_rotated": true,
        "interpolating_points": {
            "kmeans": {
                "threshold": 0.00001,
                "max_it": 1000
            }
        },
        "blacs": {
            "block_cyclic_nrows": 64,
            "block_cyclic_cocls": 64
        }
    }


thc_cfac
    Primary thc rank parameter for full basis set.
thc_half_cfac
    Thc rank for half-rotated (occupied-virtual) integrals.
theshold
    k-means convergence threshold
max_it
    k-means maximum iteration.
half_rotated
    Whether to do half-rotated THC (required for QMCPACK).
orbital_file
    File containing orbital sets.
output_file
    Output file where factorized integrals will be written.
