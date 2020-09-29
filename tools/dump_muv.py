#!/usr/bin/env python

import h5py
import matplotlib.pyplot as pl
import numpy
import sys
import scipy.linalg
from thc.transform_basis import init_from_chkfile
from pyscf.pbc.dft import numint, gen_grid
from pyscf.pbc.df.fft_ao2mo import _iskconserv, get_ao_pairs_G
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
import matplotlib.pyplot as pl
from pauxy.utils.linalg import modified_cholesky

with h5py.File(sys.argv[1], 'r') as fh5:
    aor_mu = fh5['Hamiltonian/THC/Orbitals'][:].view(numpy.complex128)
    aor_mu = aor_mu.reshape(aor_mu.shape[0], aor_mu.shape[1]).T
    iv = fh5['Hamiltonian/THC/IVecs'][:].view(numpy.complex128)
    iv = iv.reshape(iv.shape[0], iv.shape[1])

with h5py.File(sys.argv[2], 'r') as fh5:
        aor = fh5['aoR'][:]


# Compute as in pyscf using MOs, check we get the same
cell, mf, hcore, fock, AORot, kpoints, ehf_kpts, uhf = init_from_chkfile(sys.argv[3])
nb = hcore.shape[-1]

Q = kpoints[2]
shifts = []
kvecs = cell.reciprocal_vectors()
for i in range(-1,2):
    for j in range(-1,2):
        for k in range(-1,2):
            K = numpy.dot(numpy.array([i,j,k]), kvecs)
            shifts.append(K)
minus_k = []
kconserv = kpts_helper.get_kconserv(cell, kpoints)
# for ii, ki in enumerate(kpoints):
    # for ij, kj in enumerate(kpoints):
        # for G in shifts:
            # delta = kj + G
            # # print (-ki-delta, ii, ij, G)
            # if abs(numpy.dot(-ki-delta, -ki-delta)) < 1e-10:
                # minus_k.append(ij)


# print ("minus k: ", minus_k)
minus_k = numpy.zeros(len(kpoints), dtype=int)
for i in range(len(kpoints)):
    for j in range(len(kpoints)):
        if kconserv[i, 0, j] == 0:
            minus_k[i] = j
grid = cell.gen_uniform_grids()
ikpa = 0
uka = aor_mu[:,ikpa*nb:(ikpa+1)*nb]
nchol = []
gmap = numpy.zeros((len(kpoints), len(kpoints)), dtype=numpy.integer)
kmap = numpy.zeros((len(kpoints), len(kpoints)), dtype=numpy.integer)
for iq, Q in enumerate(kpoints):
    Gacs = []
    ks = []
    for ia, ka in enumerate(kpoints):
        for ic, kc in enumerate(kpoints):
            q = kc - ka
            if numpy.abs(numpy.dot(q-Q,q-Q)) < 1e-10:
                Gacs.append(numpy.array([0,0,0]))
                ks.append([ia, ic])
            else:
                q2 = q.copy()
                for G in shifts:
                    q2 = q + G
                    if numpy.abs(numpy.dot(q2-Q,q2-Q)) < 1e-10:
                        Gacs.append(G)
                        ks.append([ia,ic])

    muvs = []
    nmu = uka.shape[0]
    unique = [Gacs[0]]
    for ig, Gac in enumerate(Gacs):
        delta = [abs(numpy.dot(u-Gac,u-Gac)) > 1e-10 for u in unique]
        if all(delta):
            unique.append(Gac)
    # print(len(Gacs))
    # print (unique)
    # print (ks)
    # print(Gacs)
    for ig, Gac in enumerate(Gacs):
        kmap[iq,ig] = ks[ig][1]
        print (ks[ig][1])
        for iu, u in enumerate(unique):
            if numpy.abs(numpy.dot(Gac-u, Gac-u)) < 1e-10:
                gmap[iq,ig] = iu
    # print ("NUNIQUE: ", len(unique))
    MUV = numpy.zeros((len(unique)*nmu, len(unique)*nmu),
                      dtype=numpy.complex128)
    # print (gmap[iq])
    # print (kmap[iq])
    print (MUV.shape)
    for (iac, Gac) in enumerate(unique):
        for (ibd, Gbd) in enumerate(unique):
            # print (iac, ibd, Gac-Gbd, Q-Gac)
            coulG = numpy.sqrt(tools.get_coulG(cell, Q-Gac,
                               mesh=cell.mesh)*(cell.vol/grid.shape[0]**2))
            ivg = numpy.einsum('i,mi->mi', coulG, tools.fft(iv, cell.mesh))
            phase = numpy.exp(1j*numpy.dot(grid, Gac-Gbd))
            ivgk = numpy.einsum('i,mi->mi', coulG, tools.fft(iv*phase, cell.mesh))
            muv = numpy.dot(ivg, ivgk.conj().T)
            MUV[iac*nmu:(iac+1)*nmu,ibd*nmu:(ibd+1)*nmu] = muv
    # ikpa, ikpc = ks[0]
    # ikpb, ikpd = ks[0]
    # print (ikpa, ikpc, ikpb, ikpd)
    # iGac = 0
    # iGbd = 1
    # uka = aor_mu[:,ikpa*nb:(ikpa+1)*nb]
    # ukc = aor_mu[:,ikpc*nb:(ikpc+1)*nb]
    # ukb = aor_mu[:,ikpb*nb:(ikpb+1)*nb]
    # ukd = aor_mu[:,ikpd*nb:(ikpd+1)*nb]
    # print (numpy.max(numpy.abs(MUV-MUV.conj().T)))
    # u, s, v = scipy.linalg.svd(MUV)
    # pl.plot(s)
    # pl.yscale('log')
    # pl.show()
    L = (modified_cholesky(MUV, 1e-6, verbose=False, cmax=100).T).copy()
    print ("CHOLESKY")
    nchol.append(L.shape[1])
    # print (numpy.min(s))
    # L = scipy.linalg.cholesky(MUV, lower=True)
    # pac = numpy.einsum('ma,mc->mac', uka.conj(), ukc)
    # pdb = numpy.einsum('md,mb->mdb', ukd.conj(), ukb)
    # L = L.reshape((len(unique), len(unique), nmu, nmu))
    # # print (numpy.max(numpy.abs(numpy.dot(L[0,0],L[0,0].conj().T)-MUV)))
    # L1 = L[iGac*nmu:(iGac+1)*nmu,iGac*nmu:(iGac+1)*nmu]
    # L2 = L[iGbd*nmu:(iGbd+1)*nmu,iGac*nmu:(iGac+1)*nmu]
    # L1 = L[iGac*nmu:(iGac+1)*nmu,:]
    # L2 = L[iGbd*nmu:(iGbd+1)*nmu,:]
    # M12 = MUV[iGac*nmu:(iGac+1)*nmu,iGbd*nmu:(iGbd+1)*nmu]
    # print (numpy.max(numpy.abs(M12-numpy.dot(L1,L2.conj().T))))
    # print (numpy.max(numpy.abs(MUV-numpy.einsum('mn,pn->mp', L, L.conj()))))
    # left = numpy.einsum('mac,mn->acn', pac, L1)
    # right = numpy.einsum('mdb,mn->dbn', pdb, L2)
    # print ("ERI1")
    # eri = numpy.einsum('acn,dbn->acbd', left, right.conj())
    # print ("T1")
    # # MUV = MUV.reshape((len(unique), len(unique), nmu, nmu))
    # t1 = numpy.einsum('mik,mn->ikn', pac, M12)
    # print ("ERI2")
    # eri2 = numpy.einsum('ikn,nlj->ikjl', t1, pdb.conj())
    # # eri2 = numpy.einsum('mik,mn,nlj->ikjl', pac, MUV, pdb.conj())
    # print(eri[0,0,0,0], eri2[0,0,0,0])
    with h5py.File('integrals.h5', 'a') as fh5:
        xx = L.view(numpy.float64)
        y = numpy.zeros(L.shape, dtype=L.dtype)
        yy = y.view(numpy.float64)
        print ("reshape: ", xx.shape, L.shape, yy.shape, L.flags.c_contiguous)
        fh5['Hamiltonian/KPTHC/L%i'%iq] = xx.reshape(L.shape+(2,))

with h5py.File(sys.argv[1], 'r') as fh5:
    energies = fh5['Hamiltonian/Energies'][:]
    hcore = fh5['Hamiltonian/hcore'][:]
    occups = fh5['Hamiltonian/occups'][:]
    dims = fh5['Hamiltonian/dims'][:]
    aor_mu = fh5['Hamiltonian/THC/Orbitals'][:]

nmo = hcore.shape[0] // len(kpoints)
dims[2] = len(kpoints)
with h5py.File('integrals.h5', 'a') as fh5:
    fh5['Hamiltonian/Energies'] = energies
    for ik, k in enumerate(kpoints):
        fh5['Hamiltonian/H1_kp%i'%ik] = hcore[ik*nmo:(ik+1)*nmo,ik*nmo:(ik+1)*nmo]
    fh5['Hamiltonian/QKTok2'] = kmap
    fh5['Hamiltonian/QKToG'] = gmap
    fh5['Hamiltonian/KPoints'] = kpoints
    fh5['Hamiltonian/MinusK'] = minus_k
    fh5['Hamiltonian/NMOPerKP'] = numpy.array([nmo]*len(kpoints))
    fh5['Hamiltonian/NCholPerKP'] = numpy.array(nchol)
    fh5['Hamiltonian/dims'] = dims
    fh5['Hamiltonian/KPTHC/dims'] = numpy.array([aor_mu.shape[1], -1])
    fh5['Hamiltonian/KPTHC/Orbitals'] = aor_mu
