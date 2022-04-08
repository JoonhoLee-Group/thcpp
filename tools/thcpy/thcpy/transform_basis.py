#!/usr/bin/env python

import sys
import os
from pyscf.pbc.lib.chkfile import load_cell
from pyscf.pbc import scf
from pyscf.pbc import dft
from pyscf.pbc import tools
from pyscf import lib
import numpy as np
import time
import h5py
import scipy.linalg

from thcpy import k2gamma

def to_complex(data):
    return data.view(np.complex128).reshape(data.shape[0], data.shape[1])

def to_row_major(data):
    nr = data.shape[0]
    nc = data.shape[1]
    return data.flatten().reshape(nc, nr).T

def get_kpoint_data(h5f, name):
    groups = h5f[name]
    data = []
    ixs = [int(i) for i in list(groups.keys())]
    six = np.argsort(np.array(ixs))
    for g in groups:
        data.append(groups[g][:])
    return np.array(data)[six]

def init_from_chkfile(chkfile):
    cell = load_cell(chkfile)
    if isinstance(cell.atom, str):
        to_native_atom_fmt(cell)
    nao = cell.nao_nr()
    energy = np.asarray(lib.chkfile.load(chkfile, 'scf/e_tot'))
    kpts = np.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
    try:
        nkpts = len(kpts)
        kmf = scf.KRHF(cell, kpts)
    except:
        kmf = scf.RHF(cell)
    kmf.mo_occ = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_occ'))
    kmf.mo_coeff = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_coeff'))
    kmf.mo_energy = np.asarray(lib.chkfile.load(chkfile, 'scf/mo_energy'))
    return cell, kmf

def write_orbitals(
        kmf_super,
        kmesh,
        ao_basis=False,
        e0=0,
        filename='orbitals.h5',
        half_rotate=False,
        dtype=np.float64):
    supercell = kmf_super.cell
    grid = supercell.gen_uniform_grids()
    ngrid_points = grid.shape[0]
    print("- Number of real space grid points: {:d}".format(grid.shape[0]))
    print("- Supercell mesh: {:}".format(supercell.mesh))
    coulG = (
        tools.get_coulG(supercell, k=np.zeros(3),
                        mesh=supercell.mesh)*supercell.vol/ngrid_points**2
    )
    mo_coeff = kmf_super.mo_coeff
    hcore = kmf_super.get_hcore()
    if not ao_basis:
        hcore = mo_coeff.conj().T @ hcore @ mo_coeff
    nks = np.prod(kmesh)
    with h5py.File(filename, 'w') as fh5:
        fh5.create_dataset('real_space_grid', data=grid)
        fh5.create_dataset('kmesh', data=kmesh)
        newshape = supercell.mesh.shape + (1,)
        fh5.create_dataset('fft_grid', data=supercell.mesh.reshape(newshape))
        delta_max = np.max(np.abs(hcore-hcore.conj().T))
        if delta_max > 1e-8:
            print("WARNING: HCORE is not Hermitian. Max difference {:13.8e}.".format(delta_max))
        fh5.create_dataset('hcore', data=hcore.astype(dtype))
        fh5.create_dataset('constant_energy_factors',
                           data=np.array([e0]).reshape(1,1))
        fh5.create_dataset('num_electrons',
                           data=np.array(supercell.nelec).reshape(1,2))
        fh5.create_dataset('fft_coulomb', data=coulG.reshape(coulG.shape+(1,)))
        ngs = grid.shape[0]
        density_dset = fh5.create_dataset('density', shape=(ngs,1), dtype=np.float64)
        density_occ_dset = fh5.create_dataset('density_occ', shape=(ngs,1),
                dtype=np.float64)
        ao_dset = fh5.create_dataset('aoR', dtype=dtype,
                                     shape=(ngs,hcore.shape[-1]))
        nup = supercell.nelec[0]
        ao_half_dset = fh5.create_dataset('aoR_half', dtype=dtype,
                                          shape=(ngs,nup))
        chunk_size = 10000
        # for simplicity for now.
        if ngs < chunk_size:
            chunk_size = ngs - 1
        if ngs > chunk_size:
            num_chunks = int(ngs/chunk_size)
            start = 0
            end = chunk_size
            for chunk in range(0, num_chunks+1):
                start_time = time.time()
                if chunk % 10 == 0:
                    print(" -- Generating AO chunk {:d} of {:d}.".format(chunk+1, num_chunks+1))
                aoR = dft.numint.eval_ao(supercell, grid[start:end])
                total_time = time.time() - start_time
                if chunk % 10 == 0:
                    print(" -- AOs generated in {:f} s.".format(total_time))
                ngs_chunk = len(grid[start:end])
                rho = np.zeros((ngs_chunk,1))
                rho_occ = np.zeros((ngs_chunk,1))
                start_time = time.time()
                # Translate orthogonalising matrix to supercell basis.
                if chunk % 10 == 0:
                    print(" -- Orthogonalising AOs.")
                if not ao_basis:
                    # print(chunk, aoR.shape, mo_coeff.shape)
                    aoR = np.dot(aoR, mo_coeff)
                if half_rotate:
                    aoR_half = aoR[:,:nup].copy()
                    for ir in range(ngs_chunk):
                        rho_occ[ir] = np.dot(aoR_half[ir].conj(),aoR_half[ir]).real   # not normalized
                total_time = time.time() - start_time
                if chunk % 10 == 0:
                    print(" -- AOs orthogonalised in {:f} s.".format(total_time))
                    print(" -- Computing density.")
                for ir in range(ngs_chunk):
                    rho[ir] = np.dot(aoR[ir].conj(), aoR[ir]).real   # not normalized
                start_time = time.time()
                if chunk % 10 == 0:
                    print(" -- Dumping data to file.")
                density_dset[start:end,:] = rho
                ao_dset[start:end,:] = aoR.astype(dtype)
                if half_rotate:
                    ao_half_dset[start:end,:] = aoR_half.astype(dtype)
                    density_occ_dset[start:end,:] = rho_occ
                total_time = time.time() - start_time
                if chunk % 10 == 0:
                    print(" -- Data dumped in {:f} s.".format(total_time))
                start += chunk_size
                end += chunk_size
                if (end > ngs):
                    end = ngs
        fh5.flush()
    print("- Done ")

def write_thc_data(
        scf_chk,
        half_rotate=False,
        ao_basis=False,
        orbital_file='orbitals.h5',
        dtype=np.float64):
    cell, kmf = init_from_chkfile(scf_chk)
    kmesh = np.array(
            k2gamma.kpts_to_kmesh(cell, kmf.kpts),
            dtype=np.int32
            )
    scmf = k2gamma.k2gamma(kmf, make_real=False)
    print("Writing supercell orbitals to {:s}".format(orbital_file))
    nkpts = len(kmf.mo_occ)
    e0 = nkpts * kmf.energy_nuc()
    kpts = kmf.kpts
    e0 += -0.5 * nkpts * cell.nelectron * tools.pbc.madelung(cell, kpts)
    write_orbitals(
            scmf,
            kmesh,
            ao_basis=ao_basis,
            half_rotate=half_rotate,
            e0=e0,
            filename=orbital_file,
            dtype=dtype
            )
