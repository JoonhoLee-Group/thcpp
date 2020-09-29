#!/usr/bin/env python

import h5py
import numpy
import sys

with h5py.File(sys.argv[1]) as fh5:
    H1 = fh5["Hamiltonian/hcore"][:]
    dims = fh5["Hamiltonian/dims"]

    nmo = H1.shape[0]
    H1 = H1.view(dtype=numpy.complex128).reshape(nmo,nmo)

    cnt = 0
    for i in range(nmo):
      for j in range(i+1):
        if abs(H1[i][j])>1e-8:
          cnt += 1
    dims[0]=cnt
    Hv = numpy.zeros((cnt,2),dtype=numpy.float64)
    Hij = numpy.zeros((2*cnt),dtype=numpy.int32)
    cnt=0
    for i in range(nmo):
      for j in range(i+1):
        if abs(H1[i][j])>1e-8:
          Hv[cnt][0] = H1[i][j].real
          Hv[cnt][1] = H1[i][j].imag
          Hij[2*cnt] = i
          Hij[2*cnt+1] = j
          cnt += 1

    h5grp = fh5["Hamiltonian"]
    dummy = h5grp.create_dataset("H1", data=Hv)
    dummy = h5grp.create_dataset("H1_indx", data=Hij)
