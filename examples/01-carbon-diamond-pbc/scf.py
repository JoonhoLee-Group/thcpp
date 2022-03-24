#!/usr/bin/env python3

import sys
import os
from pyscf.pbc import scf, gto, tools
from pyscf import lib
import numpy
import time
import h5py

alat0 = 3.6
nks = 2
ngs = 12

cell = gto.Cell()
cell.a = (numpy.ones((3,3))-numpy.eye(3))*alat0/2.0
cell.atom = (('C',0,0,0),('C',numpy.array([0.25,0.25,0.25])*alat0))
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.mesh = [29,29,29]
cell.verbose = 4
cell.build()

nk = numpy.array([nks,nks,nks])
kpt = cell.make_kpts(nk)
mf = scf.KRHF(cell,kpt)
mf.chkfile = "scf.chk"
ehf = mf.kernel()

def get_ortho_ao(cell, kpts, LINDEP_CUTOFF=0):
    """Generate canonical orthogonalization transformation matrix.

    Parameters
    ----------
    cell : :class:`pyscf.pbc.cell' object.
        PBC cell object.
    kpts : :class:`numpy.array'
        List of kpoints.
    LINDEP_CUTOFF : float
        Linear dependency cutoff. Basis functions whose eigenvalues lie below
        this value are removed from the basis set. Should be set in accordance
        with value in pyscf (pyscf.scf.addons.remove_linear_dep_).

    Returns
    -------
    X : :class:`numpy.array`
        Transformation matrix.
    nmo_per_kpt : :class:`numpy.array`
        Number of OAOs orbitals per kpoint.
    """
    kpts = numpy.reshape(kpts,(-1,3))
    nkpts = len(kpts)
    nao = cell.nao_nr()
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph',hermi=1,kpts=kpts))
    X = numpy.zeros((nkpts,nao,nao),dtype=numpy.complex128)
    nmo_per_kpt = numpy.zeros(nkpts,dtype=numpy.int32)
    for k in range(nkpts):
        sdiag, Us = numpy.linalg.eigh(s1e[k])
        nmo_per_kpt[k] = sdiag[sdiag>LINDEP_CUTOFF].size
        norm = numpy.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
        X[k,:,0:nmo_per_kpt[k]] = Us[:,sdiag>LINDEP_CUTOFF] / norm
    return X, nmo_per_kpt

hcore = mf.get_hcore(kpts=kpt)                                   # obtain and store core hamiltonian
fock = (hcore + mf.get_veff(kpts=kpt))                           # store fock matrix (required with orthoAO)
LINDEP = 1e-8
X,nmo_per_kpt = get_ortho_ao(cell,kpt,LINDEP)      # store rotation to orthogonal PAO basis
with h5py.File(mf.chkfile, 'r+') as fh5:
  fh5['scf/hcore'] = hcore
  fh5['scf/fock'] = fock
  fh5['scf/orthoAORot'] = X
  fh5['scf/nmo_per_kpt'] = nmo_per_kpt
