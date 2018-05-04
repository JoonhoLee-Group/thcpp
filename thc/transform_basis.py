#!/usr/bin/env python

import sys
import os
from pyscf.pbc import scf, gto, tools, ao2mo
from pyscf.pbc.dft import gen_grid, numint
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
    hcore = numpy.asarray(lib.chkfile.load(chkfile, 'scf/hcore'))
    fock = numpy.asarray(lib.chkfile.load(chkfile, 'scf/fock'))
    energy = numpy.asarray(lib.chkfile.load(chkfile, 'scf/e_tot'))
    kpts = numpy.asarray(lib.chkfile.load(chkfile, 'scf/kpts'))
    nkpts = len(kpts)
    AORot = numpy.asarray(lib.chkfile.load(chkfile, 'scf/orthoAORot')).reshape(nkpts,nao,-1)
    # benchmark
    # construct
    kmf = scf.KRHF(cell, kpts)
    # kmf.mo_occ = numpy.asarray(lib.chkfile.load(chkfile, 'scf/mo_occ'))
    # kmf.mo_energies = numpy.asarray(lib.chkfile.load(chkfile, 'scf/mo_energy'))
    with h5py.File(chkfile, 'r') as fh5:
        try:
            kmf.mo_occ = get_kpoint_data(fh5, 'scf/mo_occ__from_list__/')
            kmf.mo_coeff = get_kpoint_data(fh5, 'scf/mo_coeff__from_list__/')
            kmf.mo_energy = get_kpoint_data(fh5, 'scf/mo_energy__from_list__/')
            if len(kmf.mo_occ) == 4:
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

def thc_vjk(P, Muv, dm, get_vk=True):
    # construct J and K matrices, note the order of indexing relative to pyscf
    # since our v_ijkl uses physics convention.
    # first J:
    # VJ_{jl} = \sum_{ik} v_{ijkl} dm_ki
    #        = \sum_{u} (\sum_v M_uv P_vjl) (\sum_{ik} P_uik d_ki)
    #        = \sum_u t1_ujl t2_u
    t1 = numpy.einsum('uv,vjl->ujl', Muv, P)
    t2 = numpy.einsum('uik,ki->u', P, dm)
    vj = numpy.einsum('ujl,u->jl', t1, t2)
    if get_vk:
        # next K:
        # VK_{il} = \sum_{kj} v_{ijkl} dm_kj
        #         = \sum_{kv} (\sum_u P_uik M_uv ) (\sum_{j} P_vjl d_kj)
        #         = \sum_{kv} t1_vik t2_vlk
        t1 = numpy.einsum('uik,uv->vik', P, Muv)
        t2 = numpy.einsum('vjl,kj->vlk', P, dm)
        vk = numpy.einsum('vik,vlk->il', t2, t1)
    else:
        vk = 0.0

    return (vj, vk)

def contract_thc_old(P, Muv, dm):
    # RHF!
    t1 = numpy.einsum('uik,ik->u', P, dm)
    with h5py.File('t1.h5', 'w') as fh5:
        fh5.create_dataset('t1', data=t1)
    ec = 2 * numpy.einsum('u,uv,v', t1, Muv, t1)
    t2 = numpy.einsum('uik,il->ukl', P, dm)
    m1 = numpy.einsum('ukl,uv->vkl', t2, Muv)
    ex = - numpy.einsum('vkl,vlk->', m1, t2)
    return (ec, ex)

def contract_thc(P, Muv, dm):
    # RHF!
    t1 = numpy.einsum('ui,ik,uk->u', P.conj(), dm, P)
    with h5py.File('t1.h5', 'w') as fh5:
        fh5.create_dataset('t1', data=t1)
    ec = 2 * numpy.einsum('u,uv,v', t1, Muv, t1)
    t2 = numpy.einsum('ui,il,vl->uv', P.conj(), dm, P)
    ex = - numpy.einsum('uv,uv,vu->', t2, Muv, t2)
    return (ec, ex)

def get_thc_data(thc_file):
    with h5py.File(thc_file, 'r') as fh5:
        # old format
        try:
            # old format
            Muv = fh5['muv'][:]
            P = fh5['phi_iu'][:]
        except KeyError:
            # new format
            try:
                Muv = fh5['Hamiltonian/THC/Muv'][:]
            except KeyError:
                Luv = fh5['Hamiltonian/THC/Luv'][:]
                norb = Luv.shape[0]
                Luv = Luv.view(numpy.complex128).reshape(norb,norb)
                Muv = numpy.dot(Luv, Luv.conj().T)
            P = fh5['Hamiltonian/THC/orbitals'][:]
            hcore = fh5['Hamiltonian/hcore'][:]
            if (len(list(fh5['Hamiltonian'].keys())) > 1):
                # QMCPACK complex format
                Muv = Muv.view(numpy.complex128).reshape((Muv.shape[0], Muv.shape[1]))
                P = P.view(numpy.complex128).reshape((P.shape[0], P.shape[1])).T
                hcore = hcore.view(numpy.complex128).reshape((hcore.shape[0], hcore.shape[1]))
    return (Muv, P, hcore)

def compute_thc_hf_energy_wfn(scf_dump, thc_data="fcidump.h5"):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    # s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    # (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals(fock, CikJ, s1e)
    (nup, ndown) = supercell.nelec
    nmo = nkpts * mf.mo_coeff.shape[-1]
    wfn = read_mo_matrix("wfn.dat").reshape(nmo,nmo)
    # assuming energy ordered.
    psi = wfn[:,:nup]
    # orthogonalised AOs.
    G = (psi.dot(psi.conj().T)).T
    (Muv, P, hcore) = get_thc_data(thc_data)
    # RHF
    e1b = 2 * numpy.einsum('ij,ij->', hcore, G)
    print ("e1b: ", e1b)
    nkpts = mf.mo_coeff.shape[0]
    enuc = nkpts * mf.energy_nuc()
    exxdiv = -0.5 * nkpts * cell.nelectron * tools.pbc.madelung(cell, kpts)
    ec, ex = contract_thc(P, Muv, G)
    ehf = (e1b + (ec+ex) + exxdiv + enuc) / nkpts
    print (e1b, ec, ex, exxdiv, enuc, nkpts)
    return (ehf.real, ehf_kpts)

def compute_potentials(scf_dump, thc_data="fcidump.h5"):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    (mo_energies, mo_orbs) = supercell_molecular_orbitals(AORot, fock, CikJ)
    (nup, ndown) = supercell.nelec
    # assuming energy ordered.
    psi = mo_orbs[:,:nup]
    # orthogonalised AOs.
    G = (psi.dot(psi.conj().T)).T
    with h5py.File(thc_data, 'r') as fh5:
        # new format
        Luv = fh5['Hamiltonian/THC/Luv'][:]
        P = fh5['Hamiltonian/THC/orbitals'][:]
    t1 = 2 * numpy.einsum('ui,ij,uj->u', P.conj(), G, P)
    vbias = numpy.einsum('uq,u->q', Luv, t1)
    s = numpy.sum(vbias)
    print ("sum(vbias): (%f, %f)"%(s.real, s.imag))
    for (i, vb) in enumerate(vbias):
        print ("%d (%f , %f)"%(i, vb.real, vb.imag))
    t2 = numpy.einsum('uq,q->u', Luv, vbias)
    vhs = numpy.einsum('ui,u,uk->ik', P.conj(), t2, P)
    s = numpy.sum(vhs)
    print ("sum(vhs): (%f, %f)"%(s.real, s.imag))
    for i in range(vhs.shape[0]):
        for j in range(vhs.shape[1]):
            print ("%d %d (%f, %f)"%(i, j, vhs[i,j].real, vhs[i,j].imag))

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
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    # assuming we have a regular 3d grid of kpoints
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    dm = mf.make_rdm1()
    dm_sc = kpoints_to_supercell(dm, CikJ)
    # Sanity checks. Check that h1e transform correctly.
    hcore_sc = kpoints_to_supercell(hcore, CikJ)
    print ("ecore: ", numpy.einsum('ij,ij->', hcore_sc, dm_sc))
    # test_arrays(shcore[0].real, hcore_sc.real)
    nao = dm_sc.shape[-1]
    # orbital products
    (Muv, P) = get_thc_data(thc_data)
    # update ~
    # P = numpy.einsum('ui,uj->uij', interp_orbs.conj(), interp_orbs)
    (vj, vk) = thc_vjk(P, Muv, dm_sc)
    # Madelung contribution contstructed from the supercell
    _ewald_exxdiv_for_G0(supercell, numpy.zeros(3), dm_sc.reshape(-1,nao,nao),
                         vk.reshape(-1,nao,nao))
    # hcore_sc = hcore.reshape((nao,nao))
    vhf = vj - 0.5 * vk
    fock = hcore_sc + vhf
    enuc = mf.energy_nuc() # per cell
    print (numpy.einsum('ij,ij->', 0.5*vj, dm_sc))
    print (numpy.einsum('ij,ij->', 0.5*vk, dm_sc))
    print (numpy.einsum('ij,ij->', hcore_sc, dm_sc), numpy.einsum('ij,ij->', 0.5*vhf, dm_sc))
    elec = numpy.einsum('ij,ij->', hcore_sc + 0.5*vhf, dm_sc)
    ehf = (elec + nkpts * enuc) / nkpts
    return (ehf.real, ehf_kpts, fock)

def get_transformed_orthoAO(S, LINDEP_CUTOFF):
    sdiag, Us = numpy.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / numpy.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X

def write_mo_matrix(out, mos, nao):
    for i in range(0, nao):
        for j in range(0, nao):
            val = mos[i,j]
            out.write('(%.10e,%.10e) '%(val.real, val.imag))
        out.write('\n')

def read_mo_matrix(filename):
    with open(filename) as f:
        content = f.readlines()[8:]
    useable = numpy.array([c.split() for c in content]).flatten()
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)

def dump_trial_wavefunction(supercell_mo_orbs, nelec, filename='wfn.dat'):
    namelist = "&FCI\n UHF = %d\n FullMO \n NCI = 1\n TYPE = matrix\n/"%(len(supercell_mo_orbs.shape)==3)
    with open(filename, 'w') as f:
        f.write(namelist+'\n')
        f.write('Coefficients: 1.0\n')
        f.write('Determinant: 1\n')
        nao = supercell_mo_orbs.shape[-1]
        if (len(supercell_mo_orbs.shape) == 3):
            nao = supercell_mo_orbs[0].shape[-1]
            write_mo_matrix(f, supercell_mo_orbs[0], nao)
            nao = supercell_mo_orbs[1].shape[-1]
            write_mo_matrix(f, supercell_mo_orbs[1], nao)
        else:
            write_mo_matrix(f, supercell_mo_orbs, nao)

def dump_orbitals(supercell, AORot, CikJ, hcore, e0=0, ortho_ao=False,
                  filename='orbitals.h5', half_rotate=False):
    grid = gen_grid.gen_uniform_grids(supercell)
    aoR = numint.eval_ao(supercell, grid)
    if ortho_ao:
        # Translate one-body hamiltonian from non-orthogonal kpoint basis to
        # orthogonal supercell basis.
        hcore = scipy.linalg.block_diag(*hcore)
        hcore = CikJ.dot(hcore).dot(CikJ.conj().T)
        if ortho_ao:
            hcore = unitary_transform(hcore, AORot)
        # Translate orthogonalising matrix to supercell basis.
        aoR = numpy.dot(aoR, AORot)
        if half_rotate:
            nup = supercell.nelec[0]
            aoR_half = numpy.dot(aoR, numpy.identity(AORot.shape[0])[:,:nup])
    else:
        hcore = scipy.linalg.block_diag(*hcore)
    ngs = grid.shape[0]
    rho = numpy.zeros((grid.shape[0],1))
    coulG = (
        tools.get_coulG(supercell, k=numpy.zeros(3),
                        gs=supercell.gs)*supercell.vol/ngs**2
    )
    for i in range(ngs):
        rho[i] = numpy.dot(aoR[i].conj(),aoR[i]).real   # not normalized
    with h5py.File(filename, 'w') as fh5:
        fh5.create_dataset('real_space_grid', data=grid)
        ao_dset = fh5.create_dataset('aoR', data=aoR.astype(numpy.complex128))
        ao_dset.attrs["orthogonalised"] = ortho_ao
        ao_dset = fh5.create_dataset('aoR_half', data=aoR.astype(numpy.complex128))
        fh5.create_dataset('density', data=rho)
        fh5.create_dataset('hcore', data=hcore)
        fh5.create_dataset('constant_energy_factors',
                           data=numpy.array([e0]).reshape(1,1))
        fh5.create_dataset('num_electrons',
                           data=numpy.array(supercell.nelec).reshape(1,2))
        fh5.create_dataset('fft_coulomb', data=coulG.reshape(coulG.shape+(1,)))
        fh5.create_dataset('AORot', data=AORot)
        fh5.flush()

def unitary_transform(A, P):
    return P.conj().T.dot(A).dot(P)

def supercell_molecular_orbitals_mo_basis(fock, CikJ, S):
    # Extend to large block matrix.
    S = scipy.linalg.block_diag(*S)
    fock = scipy.linalg.block_diag(*fock)
    # Transform to non-orthogonal supercell AOs
    S = CikJ.dot(S).dot(CikJ.conj().T)
    fock = CikJ.dot(fock).dot(CikJ.conj().T)
    (mo_energies, AORot) = scipy.linalg.eigh(fock, S)
    mo_orbs = numpy.identity(S.shape[0])
    return (mo_energies, mo_orbs, AORot)

def supercell_molecular_orbitals(fock, CikJ, S):
    # Extend to large block matrix.
    S = scipy.linalg.block_diag(*S)
    # Transform to non-orthogonal supercell AOs
    # S = unitary_transform(S, CikJ)
    S = CikJ.dot(S).dot(CikJ.conj().T)
    AORot = get_transformed_orthoAO(S, 1e-14)
    fock = scipy.linalg.block_diag(*fock)
    # Transform to non-orthogonal supercell AOs
    # fock_sc = unitary_transform(fock, CikJ)
    fock_sc = CikJ.dot(fock).dot(CikJ.conj().T)
    # Transform orthogonal supercell AOs
    ortho_fock_sc = unitary_transform(fock_sc, AORot)
    (mo_energies, mo_orbs) = scipy.linalg.eigh(ortho_fock_sc)
    return (mo_energies, mo_orbs, AORot)

def supercell_molecular_orbitals_uhf(AORot, fock, CikJ):
    # Extend to large block matrix.
    (tmp_energies, tmp_mo_orbs) = supercell_molecular_orbitals(AORot, fock[0], CikJ)
    mo_energies = numpy.zeros(shape=((2,)+tmp_energies.shape),
                              dtype=tmp_energies.dtype)
    mo_orbs = numpy.zeros(shape=((2,)+tmp_mo_orbs.shape),
                          dtype=tmp_mo_orbs.dtype)
    mo_energies[0] = numpy.copy(tmp_energies)
    mo_orbs[0] = numpy.copy(tmp_mo_orbs)
    (mo_energies[1], mo_orbs[1]) = supercell_molecular_orbitals(AORot, fock[1], CikJ)
    return (mo_energies, mo_orbs)

def dump_thc_data_sc(scf_dump, ortho_ao=False, wfn_file='wfn.dat',
                     orbital_file='orbitals.h5'):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    AORot = AORot[0]
    fock = fock[0]
    ortho_fock = unitary_transform(fock, AORot)
    (mo_energies, mo_orbs) = scipy.linalg.eigh(ortho_fock)
    dump_trial_wavefunction(mo_orbs, cell.nelec, False, filename=wfn_file)
    e0 = mf.energy_nuc()
    e0 += -0.5 * cell.nelectron * tools.pbc.madelung(cell, kpts)
    dump_orbitals(cell, AORot, None, hcore, e0=e0,
                  ortho_ao=ortho_ao, mos=False, filename=orbital_file)

def dump_thc_data(scf_dump, mos=False, ortho_ao=False, half_rotate=False,
                  wfn_file='wfn.dat', orbital_file='orbitals.h5'):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    if uhf:
        (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals_uhf(fock, CikJ, s1e)
    elif mos:
        (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals_mo_basis(fock, CikJ, s1e)
    else:
        (mo_energies, mo_orbs, AORot) = supercell_molecular_orbitals(fock, CikJ, s1e)
    # Dump wavefunction to file.
    print ("Writing trial wavefunction to %s"%wfn_file)
    dump_trial_wavefunction(mo_orbs, supercell.nelec, filename=wfn_file)
    # Dump thc data.
    print ("Writing supercell orbitals to %s"%orbital_file)
    e0 = nkpts * mf.energy_nuc()
    e0 += -0.5 * nkpts * cell.nelectron * tools.pbc.madelung(cell, kpts)
    dump_orbitals(supercell, AORot, CikJ, hcore, half_rotate=half_rotate, e0=e0,
                  ortho_ao=ortho_ao, filename=orbital_file)

def test_mos(scf_dump):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    hcore = scipy.linalg.block_diag(*hcore)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    nmo = hcore.shape[-1]
    wfn = read_mo_matrix("wfn_thc.dat").reshape(nmo,nmo)
    # assuming energy ordered.
    (nup, ndown) = supercell.nelec
    psi = wfn[:,:nup]
    G = (psi.dot(psi.conj().T)).T
    rdm = mf.make_rdm1()
    rdm = scipy.linalg.block_diag(*rdm)
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    S = scipy.linalg.block_diag(*s1e)
    print ("ecore: ", numpy.einsum('ji,ji->', hcore.conj(), rdm),
           (rdm.dot(S)).trace(), rdm.trace())
    rdm = scipy.linalg.block_diag(*rdm)
    mo_energies, mo_orbs, AORot = supercell_molecular_orbitals_mo_basis(fock, CikJ, s1e)
    hcore_sc = CikJ.dot(hcore).dot(CikJ.conj().T)
    # scmf = scf.RHF(supercell)
    # h1e_sc = scmf.get_hcore()
    # print (numpy.max(hcore_sc-h1e_sc))
    print (mo_energies-numpy.sort(mf.mo_energy.flatten()))
    hcore_mo = AORot.conj().T.dot(hcore_sc).dot(AORot)
    print ("ecore mo: ", 2*numpy.einsum('ji,ji->', hcore_mo.conj(), G),
            G.trace())

def dump_wavefunction_old(scf_dump):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    rdm = mf.make_rdm1()
    rdm = scipy.linalg.block_diag(*rdm)
    # mos = scipy.linalg.block_diag(*mf.mo_coeff)
    # mos = lib.chkfile.load(scf_dump, 'scf/mo_coeff')
    # mos = scipy.linalg.block_diag(*mf.mo_coeff)
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    S = scipy.linalg.block_diag(*s1e)
    # eigs = numpy.diag(mos.conj().T.dot(fock).dot(mos))
    # nup = supercell.nelec[0]
    # e, ev = scipy.linalg.eigh(fock, S)
    # eigs2 = numpy.diag(ev.conj().T.dot(fock).dot(ev))
    # psi = ev[:,:nup]
    # rdm2 = (psi.dot(psi.conj().T)).T
    # hcore = mf.get_hcore()
    hcore = scipy.linalg.block_diag(*hcore)
    # fock4 = hcore + mf.get_veff(dm_kpts=rdm, kpts=kpts)
    C = mf.mo_coeff
    C = scipy.linalg.block_diag(*C)
    # hmo = (C.conj().T).dot(hcore[0].dot(C))
    # rdmmo = scipy.linalg.inv(C).dot(rdm[0].dot(scipy.linalg.inv(C).conj().T))
    print (hcore.shape, rdm.shape)
    print ("ecore: ", numpy.einsum('ji,ji->', hcore.conj(), rdm),
           (rdm.dot(S)).trace(), rdm.trace())
    # print ("ecore: ", 2*numpy.einsum('ij,ij->', hcore, rdm2))
    scmf = scf.RHF(supercell)
    h1e_sc = scmf.get_hcore()
    rdm_sc = CikJ.dot(rdm).dot(CikJ.conj().T)
    S_sc = CikJ.dot(S).dot(CikJ.conj().T)
    # AORot2 = getOrthoAORotationSupercell(supercell, 1e-8)
    # h1e_sc = AORot2.conj().T.dot(h1e_sc).dot(AORot2)
    # rdm_sc = AORot2.conj().T.dot(rdm_sc).dot(AORot2)
    print (h1e_sc.shape, rdm_sc.shape)
    print ("ecore_sc: ", numpy.einsum('ji,ji->', h1e_sc.conj(), rdm_sc),
           rdm_sc.dot(S_sc).trace())
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    e1 = []
    # fock2 = numpy.copy(fock)
    # aorrot = numpy.copy(AORot)
    # fock2 = scipy.linalg.block_diag(*fock)
    # s1e = scipy.linalg.block_diag(*s1e)
    # AORot = scipy.linalg.block_diag(*AORot)
    # ortho_fock = AORot.conj().T.dot(fock2).dot(AORot)
    # e, ev = scipy.linalg.eigh(fock2, s1e)
    # e1b = 0
    # nup = cell.nelec[0]
    # nup = supercell.nelec[0]
    # ortho_fock_sc = CikJ.dot(ortho_fock).dot(CikJ.conj().T)
    # e3, ev3 = scipy.linalg.eigh(ortho_fock_sc)
    # attempt 4
    # overlap to for supercell aos
    S = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    (e,o,A) = supercell_molecular_orbitals(fock, CikJ, s1e)
    # S = scipy.linalg.block_diag(*S)
    # S = CikJ.dot(S).dot(CikJ.conj().T)
    # AORot2 = get_transformed_orthoAO(S, 1e-8)
    # In the non-orth-sc basis
    fock = scipy.linalg.block_diag(*fock)
    fock_sc = (CikJ).dot(fock).dot(CikJ.conj().T)
    # ..
    ortho_fock_sc2 = (A.conj().T).dot(fock_sc).dot(A)
    e4, ev4 = scipy.linalg.eigh(ortho_fock_sc2)
    # test_arrays(e, e2)
    # test_arrays(e, e3)
    test_arrays(e, e4)
    # # HF wavefunction in basis of orthogonalised supercell AOs.
    nup = supercell.nelec[0]
    psi = ev4[:,:nup]
    G = (psi.dot(psi.conj().T)).T
    hcore_sc = CikJ.dot(hcore).dot(CikJ.conj().T)
    print ("diff: ", numpy.max(hcore_sc-h1e_sc))
    hcore_sc_ortho = (A.conj().T).dot(hcore_sc).dot(A)
    print ("ecore_sc_trans: ", 2*numpy.einsum('ij,ij->', hcore_sc_ortho, G), G.trace())
    hcore_sc_ortho2 = (A.conj().T).dot(h1e_sc).dot(A)
    print ("ecore_sc_trans: ", 2*numpy.einsum('ij,ij->', hcore_sc_ortho2, G), G.trace())
    print (numpy.max(hcore_sc_ortho2-hcore_sc_ortho))
    # print ("ecore_sc_trans: ", 2*numpy.einsum('ij,ij->', hcore_sc_ortho, G), G.trace())
