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

def init_from_chkfile(chkfile):
  with h5py.File(chkfile) as fh5:
    mo_occ = get_kpoint_data(fh5, 'scf/mo_occ__from_list__')
    mo_coeff = get_kpoint_data(fh5, 'scf/mo_coeff__from_list__')
    # benchmark
    hcore = fh5['scf/hcore'][:]
    fock = fh5['scf/fock'][:]
    energy = fh5['scf/e_tot'][()]
    kpts = fh5['scf/kpts'][:]
    AORot = fh5['scf/orthoAORot'][:]
    # construct
    cellstr = fh5['mol'][()]
    cell = gto.cell.loads(cellstr)
    kmf = scf.KRHF(cell, kpts)
    kmf.mo_coeff = mo_coeff
    kmf.mo_occ = mo_occ
  return (cell, kmf, hcore, fock, AORot, kpts, energy)

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

def compute_thc_hf_energy_wfn(scf_dump, thc_data="fcidump.h5"):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    (mo_energies, mo_orbs) = supercell_molecular_orbitals(AORot, fock, CikJ)
    (nup, ndown) = supercell.nelec
    # assuming energy ordered.
    psi = mo_orbs[:,:nup]
    # orthogonalised AOs.
    G = 2*(psi.dot(psi.conj().T)).T # RHF!
    nkpts = mf.mo_coeff.shape[0]
    AORot = scipy.linalg.block_diag(*AORot)
    hcore = scipy.linalg.block_diag(*hcore)
    hcore = AORot.conj().T.dot(hcore).dot(AORot)
    hcore = CikJ.conj().T.dot(hcore).dot(CikJ)

    with h5py.File(thc_data, 'r') as fh5:
        # new format
        Muv = fh5['Hamiltonian/THC/Muv'][:]
        interp_orbs = fh5['Hamiltonian/THC/orbitals'][:]
    # Orbital products.
    P = numpy.einsum('ui,uj->uij', interp_orbs.conj(), interp_orbs)
    (vj, vk) = thc_vjk(P, Muv, G, True)
    vhf = vj - 0.5 * vk
    e1b = numpy.einsum('ij,ij->', hcore, G)
    print (e1b, numpy.einsum('ij,ij->', 0.5*vhf, G))
    ecoul = 0.5 * numpy.einsum('ij,ij->', vhf, G)
    enuc = nkpts * mf.energy_nuc()
    exxdiv = -0.5 * nkpts * cell.nelectron * tools.pbc.madelung(cell, kpts)
    ehf = (e1b + ecoul + exxdiv + enuc) / nkpts
    return (ehf.real, ehf_kpts)

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
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts) = init_from_chkfile(scf_dump)
    # assuming we have a regular 3d grid of kpoints
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    dm = mf.make_rdm1()
    dm_sc = kpoints_to_supercell(dm, CikJ)
    # Sanity checks. Check that h1e transform correctly.
    hcore_sc = kpoints_to_supercell(hcore, CikJ)
    # test_arrays(shcore[0].real, hcore_sc.real)
    nao = dm_sc.shape[-1]
    with h5py.File(thc_data, 'r') as fh5:
        try:
            # old format
            Muv = fh5['muv'][:]
            interp_orbs = fh5['phi_iu'][:]
        except KeyError:
            # new format
            Muv = fh5['Hamiltonian/THC/Muv'][:]
            interp_orbs = fh5['Hamiltonian/THC/orbitals'][:]
    # orbital products
    P = numpy.einsum('ui,uj->uij', interp_orbs.conj(), interp_orbs)
    (vj, vk) = thc_vjk(P, Muv, dm_sc)
    # Madelung contribution contstructed from the supercell
    _ewald_exxdiv_for_G0(supercell, numpy.zeros(3), dm_sc.reshape(-1,nao,nao),
                         vk.reshape(-1,nao,nao))
    # hcore_sc = hcore.reshape((nao,nao))
    vhf = vj - 0.5 * vk
    fock = hcore_sc + vhf
    enuc = mf.energy_nuc() # per cell
    print (numpy.einsum('ij,ij->', 0.5*vhf, dm_sc))
    print (numpy.einsum('ij,ij->', hcore_sc, dm_sc), numpy.einsum('ij,ij->', 0.5*vhf, dm_sc))
    elec = numpy.einsum('ij,ij->', hcore_sc + 0.5*vhf, dm_sc)
    ehf = (elec + nkpts * enuc) / nkpts
    return (ehf.real, ehf_kpts, fock)

def getOrthoAORotationSupercell(cell, LINDEP_CUTOFF):
    S = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph'))
    sdiag, Us = numpy.linalg.eigh(S)
    X = Us[:,sdiag>LINDEP_CUTOFF] / numpy.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X

def dump_wavefunction(supercell_mo_orbs, filename='wfn.dat'):
    namelist = "&FCI\n UHF = 0\n FullMO \n NCI = 1\n TYPE = matrix\n/"
    with open(filename, 'w') as f:
        f.write(namelist+'\n')
        f.write('Coefficients: 1.0\n')
        f.write('Determinant: 1\n')
        nao = supercell_mo_orbs.shape[-1]
        for i in range(0, nao):
            for j in range(0, nao):
                val = supercell_mo_orbs[i,j]
                f.write('(%.10e, %.10e)'%(val.real, val.imag))
            f.write('\n')

def dump_aos(supercell, AORot, CikJ, hcore, e0=0, ortho_ao=False, filename='supercell_atomic_orbitals.h5'):
    grid = gen_grid.gen_uniform_grids(supercell)
    aoR = numint.eval_ao(supercell, grid)
    if ortho_ao:
        # Translate one-body hamiltonian from non-orthogonal kpoint point basis
        # to orthogonal supercell basis.
        AORot = scipy.linalg.block_diag(*AORot)
        hcore = scipy.linalg.block_diag(*hcore)
        hcore = AORot.conj().T.dot(hcore).dot(AORot)
        hcore = CikJ.conj().T.dot(hcore).dot(CikJ)
        # Translate orthogonalising matrix to supercell basis.
        AORot = (CikJ.conj().T).dot(AORot.dot(CikJ))
        aoR = numpy.dot(aoR, AORot)
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
        ao_dset = fh5.create_dataset('aoR', data=aoR.astype(complex))
        ao_dset.attrs["orthogonalised"] = ortho_ao
        fh5.create_dataset('density', data=rho)
        fh5.create_dataset('hcore', data=hcore)
        fh5.create_dataset('constant_energy_factors',
                           data=numpy.array([e0]).reshape(1,1))
        fh5.create_dataset('fft_coulomb', data=coulG.reshape(coulG.shape+(1,)))
        fh5.flush()

def supercell_molecular_orbitals(AORot, fock, CikJ):
    # Extend to large block matrix.
    AORot = scipy.linalg.block_diag(*AORot)
    fock = scipy.linalg.block_diag(*fock)
    # Orthogonalised fock matrix with kpoints.
    ortho_fock = AORot.conj().T.dot(fock).dot(AORot)
    # Orthogonalised fock matrix with kpoints transformed to supercell basis.
    ortho_fock_sc = CikJ.conj().T.dot(ortho_fock).dot(CikJ)
    mo_energies, mo_orbs = scipy.linalg.eigh(ortho_fock_sc)
    return (mo_energies, mo_orbs)

def dump_thc_data(scf_dump, ortho_ao=False, wfn_file='wfn.dat', ao_file='supercell_atomic_orbitals.h5'):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = num_copy(nkpts)
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    (mo_energies, mo_orbs) = supercell_molecular_orbitals(AORot, fock, CikJ)
    # Dump wavefunction to file.
    print ("Writing trial wavefunction to %s"%wfn_file)
    dump_wavefunction(mo_orbs, filename=wfn_file)
    # Dump thc data.
    print ("Writing supercell AOs to %s"%ao_file)
    e0 = nkpts * mf.energy_nuc()
    e0 += -0.5 * cell.nelectron * tools.pbc.madelung(cell, kpts)
    dump_aos(supercell, AORot, CikJ, hcore, e0=e0, ortho_ao=ortho_ao, filename=ao_file)

def dump_wavefunction_old(scf_dump):
    (cell, mf, hcore, fock, AORot, kpts, ehf_kpts) = init_from_chkfile(scf_dump)
    nkpts = len(kpts)
    ncopy = int(nkpts**(1.0/3.0))
    (CikJ, supercell) = unit_cell_to_supercell(cell, kpts, ncopy)
    rdm = mf.make_rdm1()
    rdm = scipy.linalg.block_diag(*rdm)
    hcore = mf.get_hcore()
    hcore = scipy.linalg.block_diag(*hcore)
    # fock4 = hcore + mf.get_veff(dm_kpts=rdm, kpts=kpts)
    C = mf.mo_coeff
    C = scipy.linalg.block_diag(*C)
    # hmo = (C.conj().T).dot(hcore[0].dot(C))
    # rdmmo = scipy.linalg.inv(C).dot(rdm[0].dot(scipy.linalg.inv(C).conj().T))
    print (hcore.shape, rdm.shape)
    print ("ecore: ", numpy.einsum('ij,ij->', hcore, rdm))
    scmf = scf.RHF(supercell)
    h1e_sc = scmf.get_hcore()
    rdm_sc = CikJ.conj().T.dot(rdm).dot(CikJ)
    print ("ecore_sc: ", numpy.einsum('ij,ij->', h1e_sc, rdm_sc))
    # print (mf.energy_elec())
    # psi = C[:,mf.mo_occ[0]>0]
    # S = cell.pbc_intor('cint1e_ovlp_sph')
    # print (S.shape)
    # O = scipy.linalg.inv(psi.conj().T.dot(S).dot(psi))
    # print ("OVLP: ", scipy.linalg.det(O))
    # G = (psi.dot(O).dot(psi.conj().T)).T
    # print ("ecore: ", 2*numpy.einsum('ij,ij->', hcore[0], G), G.trace())
    s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
    e1 = []
    for i, k in enumerate(kpts):
        e1.append(scipy.linalg.eigh(fock[i], s1e[i])[0])
    e1 = numpy.array(e1).flatten()
    e1.sort()
    fock = scipy.linalg.block_diag(*fock)
    s1e = scipy.linalg.block_diag(*s1e)
    AORot = scipy.linalg.block_diag(*AORot)
    ortho_fock = AORot.conj().T.dot(fock).dot(AORot)
    e, ev = scipy.linalg.eigh(fock, s1e)
    e2, ev2 = scipy.linalg.eigh(ortho_fock)
    ortho_fock_sc = CikJ.conj().T.dot(ortho_fock).dot(CikJ)
    e3, ev3 = scipy.linalg.eigh(ortho_fock_sc)
    # print (e[:10])
    # print (e1[:10])
    # print (e2[:10])
    # print (e3[:10])
    # # HF wavefunction in basis of orthogonalised supercell AOs.
    # psi = ev3[:,:32]
    # O = scipy.linalg.inv(psi.conj().T.dot(psi))
    # print ("OVLP: ", scipy.linalg.det(O))
    G = (psi.dot(psi.conj().T)).T
    AORot = CikJ.conj().T.dot(AORot).dot(CikJ)
    # S = lib.asarray(supercell.pbc_intor('cint1e_ovlp_sph', hermi=1))
    # sdiag = AORot.conj().T.dot(S.dot(AORot))
    # AORot2 = getOrthoAORotationSupercell(supercell, 1e-8)
    # print (numpy.sum(AORot-AORot2))
    # hcore_ortho_ao = (AORot.conj().T).dot(h1e_sc.dot(AORot))
    # print ("ecore_sc_trans: ", 2*numpy.einsum('ij,ij->', hcore_ortho_ao, G), G.trace())
