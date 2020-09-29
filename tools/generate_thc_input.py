#!/usr/bin/env python3
import argparse
import sys
from transform_basis import dump_thc_data

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
    dump_thc_data(options.chk_file, mos=True, half_rotate=True,
                  orbital_file=options.orbitals, ortho_ao=True)

if __name__ == '__main__':

    main(sys.argv[1:])
