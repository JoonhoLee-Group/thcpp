#!/usr/bin/env python

import sys
import os
from pyscf.pbc import scf, gto, tools, ao2mo
from pyscf.pbc.gto import cell
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.lib.chkfile import load_cell
from pyscf import lib
from pyscf.gto import mole
import numpy
from mpi4py import MPI
import time
import h5py
import scipy.linalg
import ast

def to_complex(data):
    return data.view(numpy.complex128).reshape(data.shape[0], data.shape[1])

def to_row_major(data):
    nr = data.shape[0]
    nc = data.shape[1]
    return data.flatten().reshape(nc, nr).T

def test_arrays(ref, test):
    try:
        numpy.testing.assert_allclose(ref, test, atol=1e-8, rtol=1e-8)
    except:
        AssertionError
        print ("Arrays differ.")

def get_kpoint_data(h5f, name):
    groups = h5f[name]
    data = []
    ixs = [int(i) for i in list(groups.keys())]
    six = numpy.argsort(numpy.array(ixs))
    for g in groups:
        data.append(groups[g][:])
    return numpy.array(data)[six]

def num_copy(nkpts):
    nc = int(nkpts**(1.0/3.0))
    if nc**3 == nkpts:
        ncopy = nc
    elif (nc+1)**3 == nkpts:
        ncopy = nc + 1
    return ncopy

def unit_cell_to_supercell(cell, kpts, nks):
    Ts = lib.cartesian_prod((numpy.arange(nks[0]),
                             numpy.arange(nks[1]),
                             numpy.arange(nks[2])))
    a = cell.lattice_vectors()
    Ts = numpy.dot(Ts, a)
    uc_slices = mole.aoslice_by_atom(cell)
    # Might be dealing with GDF calculation where this hasn't been set.
    if cell.mesh is None:
        cell.mesh = numpy.array([2*30+1,2*30+1,2*30+1])
    supercell = tools.super_cell(cell, nks)
    if supercell.mesh[0] % 2 == 0:
        supercell.mesh[0] += 1
    if supercell.mesh[1] % 2 == 0:
        supercell.mesh[1] += 1
    if supercell.mesh[2] % 2 == 0:
        supercell.mesh[2] += 1
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
    C = C / (nks[0]*nks[1]*nks[2])**0.5
    return (C, supercell)

def to_native_atom_fmt(cell):
    atm_str = cell.atom.split()
    atoms = []
    natoms = len(atm_str) // 4
    offset = 4
    for i in range(0, natoms):
        atoms.append([atm_str[i*offset],
                      float(atm_str[i*offset+1]),
                      float(atm_str[i*offset+2]),
                      float(atm_str[i*offset+3])])
    cell.atom = atoms


def init_from_chkfile(chkfile):
    cell = load_cell(chkfile)
    if isinstance(cell.atom, str):
        to_native_atom_fmt(cell)
    nao = cell.nao_nr()
    try:
        hcore = numpy.asarray(lib.chkfile.load(chkfile, 'scf/hcore'))
        fock = numpy.asarray(lib.chkfile.load(chkfile, 'scf/fock'))
    except ValueError:
        hcore = None
        fock = None
    energy = numpy.asarray(lib.chkfile.load(chkfile, 'scf/e_tot'))
    kpts = numpy.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
    nkpts = len(kpts)
    try:
        AORot = numpy.asarray(lib.chkfile.load(chkfile, 'scf/orthoAORot')).reshape(nkpts,nao,-1)
    except ValueError:
        AORot = numpy.zeros((nkpts, nao, nao))
    # benchmark
    # construct
    kmf = scf.KRHF(cell, kpts)
    try:
        # fh5 = h5py.File(chkfile, 'r')
        # kmf.mo_occ = get_kpoint_data(fh5, 'scf/mo_occ__from_list__/')
        # kmf.mo_coeff = get_kpoint_data(fh5, 'scf/mo_coeff__from_list__/')
        # kmf.mo_energy = get_kpoint_data(fh5, 'scf/mo_energy__from_list__/')
        kmf.mo_occ = numpy.asarray(lib.chkfile.load(chkfile, 'scf/mo_occ'))
        kmf.mo_coeff = numpy.asarray(lib.chkfile.load(chkfile, 'scf/mo_coeff'))
        kmf.mo_energies = numpy.asarray(lib.chkfile.load(chkfile, 'scf/mo_energy'))
        if len(kmf.mo_coeff.shape) == 4:
            uhf = True
        else:
            uhf = False
    except KeyError:
        kmf.mo_occ = None
        kmf.mo_coeff = None
        uhf = False
    return (cell, kmf, hcore, fock, AORot, kpts, energy, uhf)

def kpoints_to_supercell(A, C):
    Ablock = scipy.linalg.block_diag(*A)
    return (C.conj().T).dot(Ablock.dot(C))

def get_transformed_orthoAO(S, LINDEP_CUTOFF):
    sdiag, Us = numpy.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / numpy.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X

def supercell_gs(nks, ngs):
    ngss = numpy.array([nks[0]*ngs + (nks[0]-1)//2,
                        nks[1]*ngs + (nks[1]-1)//2,
                        nks[2]*ngs + (nks[2]-1)//2,])
    return ngss

def dump_orbitals(supercell, AORot, CikJ, hcore, nks, e0=0, ortho_ao=False,
                  filename='orbitals.h5', half_rotate=False, ngs=None):
    if ngs is not None:
        gs = supercell_gs(nks, ngs)
        mesh = numpy.array([2*ng+1 for ng in gs])
    else:
        mesh = supercell.mesh
        gs = numpy.array([(nfft-1)/2 for nfft in mesh])
    grid = cell.gen_uniform_grids(supercell, mesh=mesh)
    ngrid_points = grid.shape[0]
    print ("- Number of real space grid points: %d"%grid.shape[0])
    coulG = (
        tools.get_coulG(supercell, k=numpy.zeros(3),
                        mesh=mesh)*supercell.vol/ngrid_points**2
    )
    if CikJ is not None:
        # Translate one-body hamiltonian from non-orthogonal kpoint basis to
        # orthogonal supercell basis.
        hcore = scipy.linalg.block_diag(*hcore)
        hcore = 0.5 * (hcore + hcore.conj().T)
        # print (numpy.abs(numpy.max(hcore - hcore.conj().T)))
        hcore = (CikJ).dot(hcore).dot(CikJ.conj().T)
        hcore = 0.5 * (hcore+hcore.conj().T)
    if ortho_ao:
        hcore = unitary_transform(hcore, AORot)
    with h5py.File(filename, 'w') as fh5:
        fh5.create_dataset('real_space_grid', data=grid)
        fh5.create_dataset('kpoint_grid', data=nks.reshape(nks.shape+(1,)))
        fh5.create_dataset('fft_grid', data=gs.reshape(gs.shape+(1,)))
        delta_max = numpy.max(numpy.abs(hcore-hcore.conj().T))
        if (delta_max > 1e-8):
            print ("WARNING: HCORE is not Hermitian. Max difference %13.8e.  Symmetrising for now."%delta_max)
        # The orthogonalising matrix seems to have large elements which leads to
        # precision issues when transforming the matrix. QMCPACK checks the
        # individual elements, which are small. It's safe to explicitly
        # hermitise the matrix.
        hcore = 0.5*(hcore+hcore.conj().T)
        fh5.create_dataset('hcore', data=hcore.astype(numpy.complex128))
        fh5.create_dataset('constant_energy_factors',
                           data=numpy.array([e0]).reshape(1,1))
        fh5.create_dataset('num_electrons',
                           data=numpy.array(supercell.nelec).reshape(1,2))
        fh5.create_dataset('fft_coulomb', data=coulG.reshape(coulG.shape+(1,)))
        fh5.create_dataset('AORot', data=AORot)
        ngs = grid.shape[0]
        density_dset = fh5.create_dataset('density', shape=(ngs,1), dtype=numpy.float64)
        density_occ_dset = fh5.create_dataset('density_occ', shape=(ngs,1), dtype=numpy.float64)
        ao_dset = fh5.create_dataset('aoR', dtype=numpy.complex128,
                                     shape=(ngs,hcore.shape[-1]))
        ao_dset.attrs["orthogonalised"] = ortho_ao
        nup = supercell.nelec[0]
        ao_half_dset = fh5.create_dataset('aoR_half', dtype=numpy.complex128,
                                          shape=(ngs,nup))
        chunk_size = 10000
        # for simplicity for now.
        if ngs < chunk_size:
            chunk_size = ngs - 1
        if ngs > chunk_size:
            num_chunks = int(ngs/chunk_size)
            start = 0
            end = chunk_size
            for i in range(0, num_chunks+1):
                start_time = time.time()
                print (" -- Generating AO chunk %d of %d."%(i+1, num_chunks+1))
                aoR = numint.eval_ao(supercell, grid[start:end])
                total_time = time.time() - start_time
                print (" -- AOs generated in %f s."%total_time)
                ngs_chunk = len(grid[start:end])
                rho = numpy.zeros((ngs_chunk,1))
                rho_occ = numpy.zeros((ngs_chunk,1))
                start_time = time.time()
                if ortho_ao:
                    print (" -- Orthogonalising AOs.")
                    # Translate orthogonalising matrix to supercell basis.
                    aoR = numpy.dot(aoR, AORot)
                    # Need to fix for other basis sets.
                    if half_rotate:
                        aoR_half = numpy.dot(aoR, numpy.identity(AORot.shape[0])[:,:nup])
                        for i in range(ngs_chunk):
                            rho_occ[i] = numpy.dot(aoR_half[i].conj(),aoR_half[i]).real   # not normalized
                total_time = time.time() - start_time
                print (" -- AOs orthogonalised in %f s."%total_time)
                print (" -- Computing density.")
                for i in range(ngs_chunk):
                    rho[i] = numpy.dot(aoR[i].conj(),aoR[i]).real   # not normalized
                start_time = time.time()
                print (" -- Dumping data to file.")
                density_dset[start:end,:] = rho
                density_occ_dset[start:end,:] = rho_occ
                ao_dset[start:end,:] = aoR.astype(numpy.complex128)
                ao_half_dset[start:end,:] = aoR_half.astype(numpy.complex128)
                total_time = time.time() - start_time
                print (" -- Data dumped in %f s."%total_time)
                start += chunk_size
                end += chunk_size
                if (end > ngs):
                    end = ngs
        fh5.flush()
    print ("- Done ")

def unitary_transform(A, P):
    return numpy.dot(P.conj().T, numpy.dot(A, P))

def molecular_orbitals_rhf(fock, AORot):
    fock_ortho = unitary_transform(fock, AORot)
    mo_energies, mo_orbs = scipy.linalg.eigh(fock_ortho)
    return (mo_energies, mo_orbs)

def molecular_orbitals_uhf(fock, AORot):
    mo_energies = numpy.zeros((2, fock.shape[-1]))
    mo_orbs = numpy.zeros((2, fock.shape[-1], fock.shape[-1]))
    fock_ortho = unitary_transform(fock[0], AORot)
    (mo_energies[0], mo_orbs[0]) = scipy.linalg.eigh(fock_ortho)
    fock_ortho = unitary_transform(fock[1], AORot)
    (mo_energies[1], mo_orbs[1]) = scipy.linalg.eigh(fock_ortho)
    return (mo_energies, mo_orbs)

def supercell_molecular_orbitals_mo_basis(fock, CikJ, S):
    # Extend to large block matrix.
    S = scipy.linalg.block_diag(*S)
    fock = scipy.linalg.block_diag(*fock)
    # Transform to non-orthogonal supercell AOs
    S = (CikJ).dot(S).dot(CikJ.conj().T)
    fock = (CikJ).dot(fock).dot(CikJ.conj().T)
    (mo_energies, AORot) = scipy.linalg.eigh(fock, S)
    mo_orbs = numpy.identity(S.shape[0])
    return (mo_energies, mo_orbs, AORot)

def supercell_molecular_orbitals(fock, CikJ, S):
    # Extend to large block matrix.
    S = scipy.linalg.block_diag(*S)
    # Transform to non-orthogonal supercell AOs
    # S = unitary_transform(S, CikJ)
    S = (CikJ.conj().T).dot(S).dot(CikJ)
    AORot = get_transformed_orthoAO(S, 1e-14)
    fock = scipy.linalg.block_diag(*fock)
    # Transform to non-orthogonal supercell AOs
    # fock_sc = unitary_transform(fock, CikJ)
    fock_sc = (CikJ.conj().T).dot(fock).dot(CikJ)
    # Transform orthogonal supercell AOs
    ortho_fock_sc = unitary_transform(fock_sc, AORot)
    (mo_energies, mo_orbs) = scipy.linalg.eigh(ortho_fock_sc)
    return (mo_energies, mo_orbs, AORot)

def supercell_molecular_orbitals_uhf(fock, CikJ, S):
    # Extend to large block matrix.
    (tmp_energies, tmp_mo_orbs, A1) = supercell_molecular_orbitals_mo_basis(fock[0], CikJ, S)
    mo_energies = numpy.zeros(shape=((2,)+tmp_energies.shape),
                              dtype=tmp_energies.dtype)
    mo_orbs = numpy.zeros(shape=((2,)+tmp_mo_orbs.shape),
                          dtype=tmp_mo_orbs.dtype)
    mo_energies[0] = numpy.copy(tmp_energies)
    mo_orbs[0] = numpy.copy(tmp_mo_orbs)
    (mo_energies[1], mo_orbs[1], A2) = supercell_molecular_orbitals_mo_basis(fock[1], CikJ, S)
    return (mo_energies, mo_orbs, A1)

def dump_thc_data(scf_dump, mos=False, ortho_ao=False, half_rotate=False,
                  wfn_file='wfn.dat', orbital_file='orbitals.h5', ngs=None,
                  kpoint_grid=None):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    if len(kpts.shape) == 1:
        kpts = numpy.reshape(kpts, (1,-1))
    nkpts = len(kpts)
    if kpoint_grid is None:
        ncopy = num_copy(nkpts)
        nks = numpy.array([ncopy]*3)
    else:
        nks = numpy.array([int(nk) for nk in kpoint_grid.split()])
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, nks)
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    if uhf:
        nmo = fock.shape[-1]
        fock = numpy.reshape(fock, (2, -1, nmo, nmo))
        (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals_uhf(fock, CikJ, s1e)
    elif mos:
        (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals_mo_basis(fock, CikJ, s1e)
    else:
        (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals(fock, CikJ, s1e)
    # Dump wavefunction to file.
    # print ("Writing trial wavefunction to %s"%wfn_file)
    # dump_trial_wavefunction(mo_orbs, supercell.nelec, filename=wfn_file)
    # Dump thc data.
    print ("Writing supercell orbitals to %s"%orbital_file)
    e0 = nkpts * mf.energy_nuc()
    e0 += -0.5 * nkpts * cell.nelectron * tools.pbc.madelung(cell, kpts)
    dump_orbitals(supercell, AORot, CikJ, hcore, nks, half_rotate=half_rotate,
                  e0=e0, ortho_ao=ortho_ao, filename=orbital_file, ngs=ngs)
