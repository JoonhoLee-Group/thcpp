import time
import numpy
from operator import itemgetter
from functools import reduce
import h5py
from mpi4py import MPI
from pyscf import lib
from pyscf.pbc import gto,scf,tools,df,ao2mo
from pyscf.pbc.dft import gen_grid,numint
from pyscf.lib.chkfile import load
from pyscf import lib
from pyscf.pbc.lib.chkfile import load_cell
import thc.utils
import thc.transform_basis as tb

class AOTHC:

    def __init__(self, supercell, ortho=False, scf_dump=None):
        self.coords = gen_grid.gen_uniform_grids(supercell)
        if ortho:
            (cell, mf, hcore, fock, AORot, kpts, ehf_kpts, uhf) = tb.init_from_chkfile(scf_dump)
            nkpts = len(kpts)
            ncopy = tb.num_copy(nkpts)
            (CikJ, supercell) = tb.unit_cell_to_supercell(cell, kpts, ncopy)
            s1e = lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))
            (mo_energies, mo_orbs, AORot) = tb.supercell_molecular_orbitals(fock, CikJ, s1e)
            self.aoR = numint.eval_ao(supercell, self.coords).dot(AORot)
        else:
            self.aoR = numint.eval_ao(supercell, self.coords)
        self.norm = supercell.vol/self.coords.shape[0]
        self.nmo = self.aoR.shape[1]
        self.rgrid_shape = numpy.array(supercell.gs)*2+1
        self.ngs = numpy.prod(self.rgrid_shape)
        assert(self.ngs == self.aoR.shape[0])
        self.overlap = supercell.pbc_intor('cint1e_ovlp_sph')
        self.cmin = 5
        self.cmax = 20
        self.rho = self.calculate_density()
        self.cell = supercell
        self.comm = MPI.COMM_WORLD
        self.kmeans = thc.utils.KMeans(self.coords, comm=self.comm)
        numpy.random.seed(7)
        if self.comm.Get_rank() == 0:
            print ("# Number of electrons: %d" % sum(supercell.nelec))
            print ("# Number of grid points per basis function: %d" % self.aoR.shape[0])
            print ("# Number of basis functions: %d" % self.aoR.shape[1])

    def calculate_density(self):
        self.t_rho = time.time()
        rho = numpy.zeros(self.ngs, dtype=numpy.float64)
        for i in range(self.ngs):
            rho[i] = numpy.dot(self.aoR[i].conj(),self.aoR[i])   # not normalized
        self.t_rho = time.time() - self.t_rho
        return rho

    def kernel(self):
        if self.comm.Get_rank() == 0:
            print ("c sum_ov max_ovlp msq_ovlp approx_eri t_kmeans t_lsq t_fft t_eeval")
        for c in range(self.cmin, self.cmax):
            self.single(c, print_header=False)

    def dump_data(self, CZt, CCt, aoR_mu, interp_points):
        with h5py.File('thc_data.h5', 'w') as h5f:
            print ("Array sizes: ", CZt.nbytes / 1e9, CCt.nbytes / 1e9)
            # transposing for ease of read for lapack routines later.
            h5f.create_dataset('CZt', data=CZt.T)
            h5f.create_dataset('CCt', data=CCt.T)
            h5f.create_dataset('aoR_mu', data=aoR_mu)
            h5f.create_dataset('interp_points',
                data=interp_points.reshape(interp_points.shape+(1,)))

    def single(self, c, print_header=False, solver='numpy'):
        nPts = c*self.nmo
        IPts = numpy.sort(numpy.random.choice(self.ngs, nPts, replace=False))
        IPts = self.kmeans.kernel(self.rho, self.coords[IPts,:].copy())
        if self.comm.Get_rank() == 0:
            for n in range(nPts-1):
                assert(IPts[n]!=IPts[n+1])

            aoR_mu = self.aoR[IPts,:].copy()
            (CZt, CCt) = self.construct_cz_matrices(IPts, aoR_mu)

            # shape (Nmu,ngs)
            t_lsq = time.time()
            IVecs = self.least_squares_solver(CCt, CZt, solver)
            t_lsq = time.time() - t_lsq

            # testing overlap
            mu_int = numpy.zeros(nPts,dtype=numpy.float64)
            aoR2 = numpy.zeros((nPts,self.nmo),dtype=numpy.float64)
            for n in range(nPts):
                mu_int[n] = sum(IVecs[n,:]) * self.norm
                aoR2[n,:] = aoR_mu[n,:]*mu_int[n]

            delta_ov = numpy.dot(aoR_mu.T, aoR2) - self.overlap
            msq_ov = numpy.sum(delta_ov**2.0)**0.5 / delta_ov.size

            # shape (Nmu,ngs)
            t_fft = time.time()
            IVecsG = tools.fft(IVecs, self.cell.gs)
            t_fft = time.time() - t_fft

            # shape (ngs,)
            coulG = (
                tools.get_coulG(self.cell, k=numpy.zeros(3),
                                gs=self.cell.gs)*self.cell.vol/self.ngs**2
            )

            IVecsG *= numpy.sqrt(coulG)

            Muv = numpy.dot(IVecsG,IVecsG.T.conj())

            t_eeval = time.time()
            psum = numpy.sum(aoR_mu, axis=1)
            prod = psum.conj()*psum
            approx_eri = numpy.einsum('m,mn,n', prod, Muv, prod)
            t_eeval = time.time() - t_eeval
            if print_header:
                print ("c sum_ov max_ovlp msq_ovlp approx_eri t_kmeans t_lsq t_fft t_eeval")
            print ("%d %.10e %.10e %.10e %.10e %f %f %f %f"%(c,
                   numpy.sum(delta_ov), numpy.max(delta_ov),
                   msq_ov, approx_eri, self.kmeans.t_kmeans, t_lsq, t_fft, t_eeval))

    def read_interpolating_vectors(self, filename):
        data = h5py.File(filename, 'r')
        return data['interpolating_vectors'][:]

    def dump_thc_data(self, muv, orbs):
        with h5py.File('thc_matrices.h5', 'w') as h5f:
            h5f.create_dataset('muv', data=muv)
            h5f.create_dataset('phi_iu', data=orbs)

    def construct_muv(self, ivecs):
        print (self.cell.gs.shape, self.cell.gs)
        ivecsG = tools.fft(ivecs, self.cell.gs)
        print ("IVG: ", numpy.sum(ivecsG))
        print ("IV: ", numpy.sum(ivecs))

        # shape (ngs,)
        coulG = (
            tools.get_coulG(self.cell, k=numpy.zeros(3),
                            gs=self.cell.gs)*self.cell.vol/self.ngs**2
        )

        ivecsG *= numpy.sqrt(coulG)
        print ("ivg_sum: ", numpy.sum(ivecsG))

        Muv = numpy.dot(ivecsG,ivecsG.T.conj())
        print ("Muv: ", numpy.sum(Muv))

        return Muv

    def construct_cz_matrices(self, IPts, aoR_mu):
        # shape (Nmu,ngs)
        CZt = numpy.dot(aoR_mu, self.aoR.conj().T)
        CZt = CZt*CZt
        # shape (Nmu,Nmu)
        CCt = CZt[:,IPts].copy()
        return (CZt, CCt)

    def least_squares_solver(self, x, y, solver='numpy'):
        if solver == 'numpy':
            IVecs, lstsq_residual, CCt_rank, CCt_singular_values = (
                    numpy.linalg.lstsq(x, y)
            )
        else:
            IVecs = numpy.dot(numpy.linalg.pinv(x), y)
        return IVecs
