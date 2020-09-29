#!/usr/bin/env python

import h5py
import sys

with h5py.File(sys.argv[1], 'r') as fh5:
    Luv = fh5['Hamiltonian/THC/Luv'][:]
    Orbs = fh5['Hamiltonian/THC/orbitals'][:]
with h5py.File(sys.argv[2], 'r+') as fh5:
    del fh5['Hamiltonian/THC/Luv']
    fh5['Hamiltonian/THC/Luv'] = Luv
    del fh5['Hamiltonian/THC/Orbitals']
    fh5['Hamiltonian/THC/Orbitals'] = Orbs
    dims = fh5['Hamiltonian/THC/dims'][:]
    del fh5['Hamiltonian/THC/dims']
    dims[1] = Luv.shape[0]
    fh5['Hamiltonian/THC/dims'] = dims
