#!/usr/bin/env python

import h5py
import numpy
from pyscf import lib
from pyscf.pbc import gto, scf, tools

alat0 = 3.6

cell = gto.Cell()
cell.a = (numpy.ones((3,3))-numpy.eye(3))*alat0/2.0
cell.atom = (('C',0,0,0),('C',numpy.array([0.25,0.25,0.25])*alat0))
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.gs = [10]*3  # 10 grids on postive x direction, => 21^3 grids in total
cell.verbose = 5 
cell.build()

nk = [1,1,1]
kpt = cell.make_kpts(nk)

mf = scf.RHF(cell,kpt)
mf.chkfile = "scf.dump"
ehf = mf.kernel()

