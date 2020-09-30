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

The input file is self explanatory:

.. literalinclude:: input.json
