#!/usr/bin/env python3
import argparse
import sys
from thcpy.transform_basis import write_thc_data
import numpy as np

def parse_args(args):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-i', '--input', dest='chk_file', type=str,
                        default=None, help='Input pyscf .chk file.')
    parser.add_argument('-o', '--output', dest='orbitals',
                        type=str, default='orbitals.h5',
                        help='Output file name for QMCPACK hamiltonian.')
    options = parser.parse_args(args)

    if not options.chk_file:
        parser.print_help()
        sys.exit()

    return options

def main(args):
    """Generate THCPP input from pyscf checkpoint file.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args(args)
    write_thc_data(
            options.chk_file,
            half_rotate=True,
            dtype=np.complex128,
            orbital_file=options.orbitals,
            )

if __name__ == '__main__':

    main(sys.argv[1:])
