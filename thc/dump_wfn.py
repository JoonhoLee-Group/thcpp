#!/usr/bin/env python

from thc.transform_basis import dump_thc_data, dump_thc_data_sc
import sys
import argparse

def parse_args(args):
    '''Parse command-line arguments.

Parameters
----------
args : list of strings
    command-line arguments.

Returns
-------
args : :class:`ArgumentParser`
    Arguments read in from command line.
'''

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('-o', '--ortho', action='store_true',
                        default=False, help='Orthogonalise orbitals.')
    parser.add_argument('-m', '--mos', action='store_true',
                        default=False, help='Transform to MO basis.')
    parser.add_argument('-t', '--trial', action='store_true',
                        default=False, help='Half rotate orbitals by trial wavefunction.')
    parser.add_argument('-s', '--supercell', action='store_true',
                        default=False, help='Working with supercell SCF dump.')
    parser.add_argument('-w', '--wfn-file', dest='wfn_file', type=str,
                        default='wfn_thc.dat', help='Wavefunction file.')
    parser.add_argument('-d', '--dump-file', type=str, dest='dump_file',
                        default='aos.h5', help='Output file to write orbitals to for THC++.')
    parser.add_argument('-n', '--ngs', type=int, dest='ngs',
                        default=None, help='Real space grid.')
    parser.add_argument('-k', '--kpoints', type=str, dest='kpoints',
                        default=None, help='Grid integers.')
    parser.add_argument('-f', '--file', type=str, dest='filename', help='SCF dump.')
    args = parser.parse_args(args)

    return args

def main(args):
    args = parse_args(args)
    if args.supercell:
        dump_thc_data_sc(args.filename, ortho_ao=args.ortho,
                         orbital_file="aos.h5", wfn_file="wfn_thc.dat")
    else:
        dump_thc_data(args.filename,
                      mos=args.mos,
                      ortho_ao=args.ortho,
                      orbital_file=args.dump_file,
                      wfn_file=args.wfn_file,
                      half_rotate=args.trial,
                      ngs=args.ngs,
                      kpoint_grid=args.kpoints)

if __name__ == '__main__':
    main(sys.argv[1:])
