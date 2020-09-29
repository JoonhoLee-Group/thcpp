#!/usr/bin/env python

import sys
import os
from pyscf.pbc import gto, tools
from pyscf.pbc.dft import gen_grid, numint
import numpy
import thc.atomic_orbitals
import thc.interp_kmeans
from mpi4py import MPI
import time
import h5py

alat0 = 3.6

cell = gto.Cell()
cell.a = (numpy.ones((3,3))-numpy.eye(3))*alat0/2.0
cell.atom = (('C',0,0,0),('C',numpy.array([0.25,0.25,0.25])*alat0))
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.gs = [12]*3  # 10 grids on postive x direction, => 21^3 grids in total
cell.verbose = 5
cell.build()

ncopy = int(sys.argv[1])
c = int(sys.argv[2])

supercell = tools.super_cell(cell, numpy.array([ncopy,ncopy,ncopy]))
aothc = thc.atomic_orbitals.AOTHC(supercell)
npts = c * aothc.nmo
comm = MPI.COMM_WORLD
if comm.rank == 0:
    print ("Reading interpolating vectors.")
    ivecs = aothc.read_interpolating_vectors('thc_interpolating_vectors.h5')
    # hack
    data = h5py.File('thc_data.h5')
    aoR_mu = data['aoR_mu'][:]
    print ("Constructing muv.")
    muv = aothc.construct_muv(ivecs)
    print ("Dumping muv.")
    aothc.dump_thc_data(muv, aoR_mu)
