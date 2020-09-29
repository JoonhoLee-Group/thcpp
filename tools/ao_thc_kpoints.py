#!/usr/bin/env python

import time
import numpy
from operator import itemgetter
from functools import reduce
from pyscf import lib
from pyscf.pbc import gto,scf,tools,df,ao2mo
from pyscf.pbc.dft import gen_grid,numint
from pyscf.lib.chkfile import load
from pyscf.pbc.lib.chkfile import load_cell
from qmctools.orthoAO import getOrthoAORotation

from pyscftools.interpolating_points_k_means import IPts_k_means 

alat0 = 3.6
cell = gto.Cell()
cell.a = (numpy.ones((3,3)) - numpy.eye(3))*alat0/2.0
cell.atom = (('C',0,0,0),('C',numpy.array([0.25,0.25,0.25])*alat0))
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.gs = [10]*3
cell.verbose = 5
cell.build()
nk = [2,2,2]
cell.make_kpts(nk)

coords = gen_grid.gen_uniform_grids(cell)
norm = cell.vol/coords.shape[0]

aoR = numint.eval_ao(cell,coords)
nmo = aoR.shape[1]
print (aoR.shape)
# X, nmo = getOrthoAORotation(cell, numpy.zeros(3), 1e-5)
# aoR = aoR.dot(X)

rgrid_shape = numpy.array(cell.gs)*2+1
print ("nao_nr", cell.nao_nr())

ao_eris = ao2mo.get_ao_eri(cell)
print ("ao_eris", ao_eris.shape)
exact = numpy.sum(ao_eris)

ngs = numpy.prod(rgrid_shape)

assert ngs == aoR.shape[0]

# shape (Nmu)
# for c in range(5,20):
    # nPts = c*nmo 
    # IPts = numpy.sort(numpy.random.choice(ngs,nPts,replace=False)) 
    # rho = numpy.zeros(ngs,dtype=numpy.float64)
    # for i in range(ngs):
        # rho[i] = numpy.dot(aoR[i].conj(),aoR[i])   # not normalized
    # IPts = IPts_k_means(coords, rho, coords[IPts,:].copy(), maxIT=100, thres=1e-6)

    # for n in range(nPts-1):
        # assert(IPts[n]!=IPts[n+1])

    # # shape (Nmu,nmo)
    # aoR_mu = aoR[IPts,:].copy() 

    # # shape (Nmu,ngs)
    # CZt = numpy.dot(aoR_mu,aoR.T)
    # CZt = CZt*CZt

    # # shape (Nmu,Nmu)
    # CCt = CZt[:,IPts].copy()

    # # shape (Nmu,ngs)
    # IVecs, lstsq_residual, CCt_rank, CCt_singular_values = numpy.linalg.lstsq(CCt,CZt)
# #    print "Rank of CC^T: ",CCt_rank

    # # testing overlap
    # mu_int = numpy.zeros(nPts,dtype=numpy.float64)
    # aoR2 = numpy.zeros((nPts,nmo),dtype=numpy.float64)
    # for n in range(nPts):
        # mu_int[n] = sum(IVecs[n,:])*norm
        # aoR2[n,:] = aoR_mu[n,:]*mu_int[n]

    # ov = numpy.dot(aoR_mu.T,aoR2) - numpy.eye(nmo)
# #    print "Sum/Max overlap erros:",numpy.sum(ov),numpy.max(ov) 

    # # shape (Nmu,ngs)
    # IVecsG = tools.fft(IVecs, cell.gs)

    # # shape (ngs,)
    # coulG = tools.get_coulG(cell, k=numpy.zeros(3), gs=cell.gs)*cell.vol/ngs**2

    # IVecsG *= numpy.sqrt(coulG)

    # Muv = numpy.dot(IVecsG,IVecsG.T.conj())

    # # m1 = numpy.sum(aoR_mu.conj(), axis=1)
    # # m2 = numpy.sum(aoR_mu, axis=1)
    # # Puv = numpy.dot(aoR_mu.conj(), aoR_mu.T)
    # # approx = reduce(numpy.dot, (numpy.diagonal(Puv), Muv, numpy.diagonal(Puv).T))
    # # approx_eri = numpy.einsum('mi,mj,mn,nk,nl->ijkl', aoR_mu.conj(), aoR_mu, Muv, aoR_mu.conj(), aoR_mu) 
    # # approx = numpy.sum(approx_eri)
    # # approx2 = numpy.einsum('mi,mj,mn,nk,nl->', aoR_mu.conj(), aoR_mu, Muv, aoR_mu.conj(), aoR_mu)
    # # t1 = time.time()
    # # P = numpy.einsum('mi,mj->mij', aoR_mu.conj(), aoR_mu)
    # # eri00 = numpy.einsum('mij,mn,nkl->', P, Muv, P)
    # # t1 = time.time() - t1
    # # t2 = time.time()
    # # Psum2 = numpy.einsum('mi,mj->m', aoR_mu.conj(), aoR_mu) 
    # # eri22 = numpy.einsum('m,mn,n->', Psum2, Muv, Psum2)
    # # t2 = time.time() - t2
    # t3 = time.time()
    # psum3 = numpy.sum(aoR_mu, axis=1) 
    # prod = psum3.conj()*psum3
    # approx_eri = numpy.einsum('m,mn,n', prod, Muv, prod)
    # t3 = time.time() - t3
    # # eri2 = numpy.einsum('mi,mj,mn,nk,nl->ijkl', phi0, phi0.conj(), Muv, phi0,
            # # phi0.conj())
    # print c,CCt_rank,numpy.sum(ov),numpy.max(ov), exact, approx_eri, abs(exact-approx_eri), t3

    # # Ecoul = sum_u,v,i,j,k,l moR_mu[u,i].conj() * moR_mu[u,k] * G[i,k] * Muv * moR_mu[v,j].conj() * moR_mu[v,l] * G[j,l]
    # # Guv = reduce( numpy.dot, (moR_mu.conj(), G, moR_mu.T))
    # # Ecoul = reduce( numpy.dot, (numpy.diagonal(Guv), Muv, numpy.diagonal(Guv).T))
    # # print c,cct_rank,numpy.sum(ov),numpy.max(ov),0.5*ecoul,0.5*ecoul-1.4675819170323441
