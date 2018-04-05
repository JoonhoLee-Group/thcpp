#!/usr/bin/env python

import sys
import os
from pyscf.pbc import gto, tools
from pyscf.pbc.dft import gen_grid, numint
import numpy
import thc.atomic_orbitals
from mpi4py import MPI
import time
import h5py

ncopy = int(sys.argv[1])
c = int(sys.argv[2])
ngs = int(sys.argv[3])
alat0 = 3.6

cell = gto.Cell()
cell.a = (numpy.ones((3,3))-numpy.eye(3))*alat0/2.0
cell.atom = (('C',0,0,0),('C',numpy.array([0.25,0.25,0.25])*alat0))
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.gs = [ngs]*3  # 10 grids on postive x direction, => 21^3 grids in total
cell.verbose = 5
cell.build()


supercell = tools.super_cell(cell, numpy.array([ncopy,ncopy,ncopy]))
aothc = thc.atomic_orbitals.AOTHC(supercell)
comm = MPI.COMM_WORLD
if comm.rank == 0:
    print ("dimension of aoR: ", aothc.aoR.nbytes/1e9)
npts = c * aothc.nmo
init = numpy.sort(numpy.random.choice(aothc.ngs, npts, replace=False))
interp_points = aothc.kmeans.kernel(aothc.rho, aothc.coords[init,:].copy())
if comm.rank == 0:
    aoR_mu = aothc.aoR[interp_points,:].copy()
    (CZt, CCt) = aothc.construct_cz_matrices(interp_points, aoR_mu)
    aothc.dump_data(CZt, CCt, aoR_mu)
    print ("SUM: ", numpy.sum(CZt), numpy.sum(CCt))
    print (CZt.shape, CCt.shape, (numpy.abs(CZt) < 1e-8).sum()/(CZt.size),
           (numpy.abs(CCt) < 1e-8).sum()/CCt.size)
    print (numpy.sum(CCt.dot(CZt)))
