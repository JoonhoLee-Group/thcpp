#!/usr/bin/env python
import numpy
from operator import itemgetter
from functools import reduce
from pyscf import lib
from pyscf.pbc import gto,scf,tools,df
from pyscf.pbc.dft import gen_grid,numint
from pyscf.lib.chkfile import load
from pyscf.pbc.lib.chkfile import load_cell

from pyscftools.interpolating_points_k_means import IPts_k_means 

chkfile = 'scf.dump'

cell = load_cell(chkfile)
nao = cell.nao_nr()
kpt = lib.chkfile.load(chkfile, 'scf/kpt')

mo_occ = lib.chkfile.load(chkfile, 'scf/mo_occ')
nmo=mo_occ.shape[0]

mo = lib.chkfile.load(chkfile, 'scf/mo_coeff')
nup = 4
ndown=4
nel=nup+ndown

coords = gen_grid.gen_uniform_grids(cell)
norm = cell.vol/coords.shape[0]

aoR    = numint.eval_ao(cell,coords)

rgrid_shape = numpy.array(cell.gs)*2+1

ngs = numpy.prod(rgrid_shape)

assert ngs == aoR.shape[0]

# shape (ngs,nmo)
moR = numpy.dot(aoR,mo)

G = numpy.zeros((nmo,nmo),dtype=numpy.float64)
for i in range(nup):
    G[i,i]=2.0

# shape (Nmu)
for c in range(5,20):
    nPts = c*nmo 
    IPts = numpy.sort(numpy.random.choice(ngs,nPts,replace=False)) 
    rho = numpy.zeros(ngs,dtype=numpy.float64)
    for i in range(ngs):
        rho[i] = numpy.dot(moR[i].conj(),moR[i])   # not normalized
    IPts = IPts_k_means(coords, rho, coords[IPts,:].copy(), maxIT=100, thres=1e-6)

    for n in range(nPts-1):
        assert(IPts[n]!=IPts[n+1])

    # shape (Nmu,nmo)
    moR_mu = moR[IPts,:].copy() 

    # shape (Nmu,ngs)
    CZt = numpy.dot(moR_mu,moR.T)
    CZt = CZt*CZt

    # shape (Nmu,Nmu)
    CCt = CZt[:,IPts].copy()

    # shape (Nmu,ngs)
    IVecs, lstsq_residual, CCt_rank, CCt_singular_values = numpy.linalg.lstsq(CCt,CZt)
#    print "Rank of CC^T: ",CCt_rank

    # testing overlap
    mu_int = numpy.zeros(nPts,dtype=numpy.float64)
    moR2 = numpy.zeros((nPts,nmo),dtype=numpy.float64)
    for n in range(nPts):
        mu_int[n] = sum(IVecs[n,:])*norm
        moR2[n,:] = moR_mu[n,:]*mu_int[n]

    ov = numpy.dot(moR_mu.T,moR2) - numpy.eye(nmo)
#    print "Sum/Max overlap erros:",numpy.sum(ov),numpy.max(ov) 

    # shape (Nmu,ngs)
    IVecsG = tools.fft(IVecs, cell.gs)

    # shape (ngs,)
    coulG = tools.get_coulG(cell, k=numpy.zeros(3), gs=cell.gs)*cell.vol/ngs**2

    IVecsG *= numpy.sqrt(coulG)

    Muv = numpy.dot(IVecsG,IVecsG.T.conj())

    # Ecoul = sum_u,v,i,j,k,l moR_mu[u,i].conj() * moR_mu[u,k] * G[i,k] * Muv * moR_mu[v,j].conj() * moR_mu[v,l] * G[j,l]
    Guv = reduce( numpy.dot, (moR_mu.conj(), G, moR_mu.T))
    Ecoul = reduce( numpy.dot, (numpy.diagonal(Guv), Muv, numpy.diagonal(Guv).T))
    print c,CCt_rank,numpy.sum(ov),numpy.max(ov),0.5*Ecoul,0.5*Ecoul-1.4675819170323441
