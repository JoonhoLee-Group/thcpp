#!/usr/bin/env python

import sys
import os
from pyscf.pbc import scf, gto, tools, ao2mo
from pyscf.pbc.dft import gen_grid, numint
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf import lib
from pyscf.gto import mole
import numpy
from mpi4py import MPI
import time
import h5py
import scipy.linalg

def test_arrays(ref, test):
    try:
        numpy.testing.assert_allclose(ref, test, atol=1e-8, rtol=1e-8)
    except:
        AssertionError
        print ("Arrays differ.")

def get_kpoint_data(h5f, name):
    groups = h5f[name]
    data = []
    for g in groups:
        data.append(groups[g][:])
    return numpy.array(data)

def unit_cell_to_supercell(cell, kpts, ncopy):
    Ts = lib.cartesian_prod((numpy.arange(ncopy),
                             numpy.arange(ncopy),
                             numpy.arange(ncopy)))
    a = cell.lattice_vectors()
    Ts = numpy.dot(Ts, a)
    uc_slices = mole.aoslice_by_atom(cell)
    supercell = tools.super_cell(cell, numpy.array([ncopy,ncopy,ncopy]))
    sc_slices = mole.aoslice_by_atom(supercell)
    # transformation matrix between unit and super cell
    nbasis = sc_slices[supercell.natm-1,3]
    C = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
    offset = 0
    for (i, k) in enumerate(kpts):
        for ia, a in enumerate(cell.atom):
            iks = uc_slices[ia, 2] + offset
            ike = uc_slices[ia, 3] + offset
            ii = numpy.arange(iks, ike)
            # print ("IKS, IKE: ", iks, ike)
            for (j, T) in enumerate(Ts):
                # Super cell atoms related to atom "a" in the unit cell by a
                # translation vector.
                JS = sc_slices[cell.natm*j+ia, 2]
                JE = sc_slices[cell.natm*j+ia, 3]
                JJ = numpy.arange(JS, JE)
                C[ii,JJ] = numpy.exp(1j*k.dot(T))
        offset += uc_slices[cell.natm-1, 3]

    # assuming cubic regular grid for supercell / kpoints.
    C = C / (ncopy**3.0)**0.5
    return (C, supercell)

def init_from_chkfile(chkfile):
    fh5 = h5py.File(chkfile)
    mo_occ = get_kpoint_data(fh5, 'scf/mo_occ__from_list__')
    mo_coeff = get_kpoint_data(fh5, 'scf/mo_coeff__from_list__')
    # benchmark
    hcore = fh5['scf/hcore'][:]
    fock = fh5['scf/fock'][:]
    energy = fh5['scf/e_tot'][()]
    kpts = fh5['scf/kpts'][:]
    # construct
    cellstr = fh5['mol'][()]
    cell = gto.cell.loads(cellstr)
    kmf = scf.KRHF(cell, kpts)
    kmf.mo_coeff = mo_coeff
    kmf.mo_occ = mo_occ
    return (cell, kmf, hcore, fock, kpts, energy)

def kpoints_to_supercell(A, C):
    Ablock = scipy.linalg.block_diag(*A)
    return (C.conj().T).dot(Ablock.dot(C))

def compute_thc_hf_energy(scf_dump, thc_data='thc_matrices.h5'):
    """Compute HF energy using THC approximation to ERIs.

    Parameters
    ----------
    scf_dump : string
        pyscf chkfile hdf5 dump from kpoint calculation corresponding to
        supercell used for THC calculation.
    thc_data : string
        THC hdf5 dump containing Muv matrices and orbitals at interpolating
        points.

    Returns
    -------
    ehf : float
        HF energy per unit cell computed using THC ERIs.
    ehf_ref : float
        Reference HF energy from kpoint calculation.
    fock : :class:`numpy.ndarray`
        Fock matrix calculating using THC ERIs.
    """
    # transformation matrix
    (cell, mf, hcore, fock, kpts, ehf_kpts) = init_from_chkfile(scf_dump)
    # assuming we have a regular 3d grid of kpoints
    nkpts = len(kpts)
    ncopy = int(nkpts**(1.0/3.0))
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    dm = mf.make_rdm1()
    dm_sc = kpoints_to_supercell(dm, CikJ)
    # Sanity checks. Check that h1e transform correctly.
    hcore_sc = kpoints_to_supercell(hcore, CikJ)
    # test_arrays(shcore[0].real, hcore_sc.real)
    nao = dm_sc.shape[-1]
    with h5py.File(thc_data, 'r') as fh5:
        Muv = fh5['muv'][:]
        interp_orbs = fh5['phi_iu'][:]
    # orbital products
    P = numpy.einsum('ui,uj->uij', interp_orbs.conj(), interp_orbs)
    # construct J and K matrices, note the order of indexing relative to pyscf
    # since our v_ijkl uses physics convention.
    # first J:
    # VJ_{jl} = \sum_{ik} v_{ijkl} dm_ki
    #        = \sum_{u} (\sum_v M_uv P_vjl) (\sum_{ik} P_uik d_ki)
    #        = \sum_u t1_ujl t2_u
    t1 = numpy.einsum('uv,vjl->ujl', Muv, P)
    t2 = numpy.einsum('uik,ki->u', P, dm_sc)
    vj = numpy.einsum('ujl,u->jl', t1, t2)
    # next K:
    # VK_{il} = \sum_{kj} v_{ijkl} dm_kj
    #         = \sum_{kv} (\sum_u P_uik M_uv ) (\sum_{j} P_vjl d_kj)
    #         = \sum_{kv} t1_vik t2_vlk
    t1 = numpy.einsum('uik,uv->vik', P, Muv)
    t2 = numpy.einsum('vjl,kj->vlk', P, dm_sc)
    vk = numpy.einsum('vik,vlk->il', t2, t1)
    # Madelung contribution contstructed from the supercell
    _ewald_exxdiv_for_G0(supercell, numpy.zeros(3), dm_sc.reshape(-1,nao,nao),
                         vk.reshape(-1,nao,nao))
    # hcore_sc = hcore.reshape((nao,nao))
    vhf = vj - 0.5 * vk
    fock = hcore_sc + vhf
    enuc = mf.energy_nuc() # per cell
    elec = numpy.einsum('ij,ij->', hcore_sc + 0.5*vhf, dm_sc)
    ehf = (elec + nkpts * enuc) / nkpts
    return (ehf, ehf_kpts, fock)
