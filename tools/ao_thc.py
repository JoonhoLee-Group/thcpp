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

def ecoul(cell):
    ao_eris = ao2mo.get_ao_eri(cell)
    return numpy.sum(ao_eris)

class AOTHC:

    def __init__(self, supercell)
        self.coords = gen_grid.gen_uniform_grids(supercell)
        self.aoR = numint.eval_ao(supercell, coords)
        print ("# Number of electrons: %d" % sum(supercell.nelec))
        print ("# Number of grid points per basis function: %d" % self.aoR.shape[0])
        print ("# Number of basis functions: %d" % self.aoR.shape[1])
        self.norm = supercell.vol/coords.shape[0]
        self.nmo = aoR.shape[1]
        self.rgrid_shape = numpy.array(supercell.gs)*2+1
        self.ngs = numpy.prod(rgrid_shape)
        assert ngs == aoR.shape[0]
        self.overlap = supercell.pbc_intor('cint1e_ovlp_sph') 
        self.cmin = 5
        self.cmax = 20
        self.rho = numpy.zeros(self.ngs, dtype=numpy.float64)

    def calculate_density(self):
        self.t_rho = time.time()
        for i in range(ngs):
            self.rho[i] = numpy.dot(self.aoR[i].conj(),self.aoR[i])   # not normalized
        self.t_rho = time.time() - t_rho

    def converge(self):
        print ("c CCt_rank sum_ov max_ovlp approx_eri t_rho t_kmeans t_lsq t_fft t_eeval")
        for c in range(self.cmin, self.cmax):
            nPts = c*self.nmo 
            IPts = numpy.sort(numpy.random.choice(self.ngs,nPts,replace=False)) 
            t_kmeans = time.time()
            IPts = IPts_k_means(coords, rho, coords[IPts,:].copy(), maxIT=100, thres=1e-6)
            t_kmeans = time.time() - t_kmeans

            for n in range(nPts-1):
                assert(IPts[n]!=IPts[n+1])

            # shape (Nmu,nmo)
            aoR_mu = aoR[IPts,:].copy() 

            # shape (Nmu,ngs)
            CZt = numpy.dot(aoR_mu,aoR.T)
            CZt = CZt*CZt

            # shape (Nmu,Nmu)
            CCt = CZt[:,IPts].copy()

            # shape (Nmu,ngs)
            t_lsq = time.time()
            IVecs, lstsq_residual, CCt_rank, CCt_singular_values = numpy.linalg.lstsq(CCt,CZt)
            t_lsq = time.time() - t_lsq

            # testing overlap
            mu_int = numpy.zeros(nPts,dtype=numpy.float64)
            aoR2 = numpy.zeros((nPts,nmo),dtype=numpy.float64)
            for n in range(nPts):
                mu_int[n] = sum(IVecs[n,:])*norm
                aoR2[n,:] = aoR_mu[n,:]*mu_int[n]

            ov = numpy.dot(aoR_mu.T,aoR2) - overlap 

            # shape (Nmu,ngs)
            t_fft = time.time()
            IVecsG = tools.fft(IVecs, supercell.gs)
            t_fft = time.time() - t_fft

            # shape (ngs,)
            coulG = tools.get_coulG(supercell, k=numpy.zeros(3), gs=supercell.gs)*supercell.vol/ngs**2

            IVecsG *= numpy.sqrt(coulG)

            Muv = numpy.dot(IVecsG,IVecsG.T.conj())
            
            t_eeval = time.time()
            psum = numpy.sum(aoR_mu, axis=1) 
            prod = psum.conj()*psum
            approx_eri = numpy.einsum('m,mn,n', prod, Muv, prod)
            t_eeval = time.time() - t_eeval
            print ("%d %d %f %f %f %f %f %f %f %f"%(c, CCt_rank, numpy.sum(ov), numpy.max(ov),
                   approx_eri, t_rho, t_kmeans, t_lsq, t_fft, t_eeval))
