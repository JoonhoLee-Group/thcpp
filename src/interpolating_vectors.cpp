#include <iostream>
#include <mpi.h>
#include <time.h>
#include <fftw3.h>
#include <complex>
#include "json.hpp"
#include "interpolating_vectors.h"
#include "h5helper.h"
#include "scalapack_defs.h"
#include "context_handler.h"
#include "matrix_operations.h"
#include "utils.h"
#include "logging.h"

namespace InterpolatingVectors
{
  IVecs::IVecs(nlohmann::json &input, ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs,
               bool rotate, bool append)
  {
    int filename_size;
    if (BH.rank == 0) {
      input_file = input.at("orbital_file").get<std::string>();
      filename_size = input_file.size();
      output_file = input.at("output_file").get<std::string>();
      std::cout << "#################################################" << std::endl;
      std::cout << "##   Setting up interpolative vector solver.   ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      std::vector<hsize_t> dims(2);
      H5Helper::read_dims(input_file, "aoR", dims);
      nbasis = dims[1];
      if (append) H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_TRUNC);
    }
    MPI_Bcast(&nbasis, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (BH.rank != 0) input_file.resize(filename_size);
    MPI_Bcast(&input_file[0], input_file.size()+1, MPI_CHAR, 0, MPI_COMM_WORLD);
    DistributedMatrix::Matrix<int> fft(input_file, "fft_grid", BH.Root);
    if (BH.rank == 0) {
      fft_grid = fft.store;
    } else {
      fft_grid.resize(3);
    }
    MPI_Bcast(fft_grid.data(), fft_grid.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if (rotate) {
      // to distinguish Luv between half rotated and rotated case
      prefix = "HalfTransformed";
      setup_CZt_half(interp_indxs, BH);
      half_rotate = true;
    } else {
      setup_CZt(interp_indxs, BH);
      prefix = "";
      half_rotate = false;
    }
    setup_CCt(interp_indxs, BH);
    check_rank(BH);
  }

  void IVecs::setup_pseudo_dm(DistributedMatrix::Matrix<std::complex<double> > &Pua,
                              std::vector<int> &interp_indxs,
                              ContextHandler::BlacsHandler &BH,
                              std::string aos, bool write,
                              std::string prfx="")
  {
    // (M, Ngrid).
    DistributedMatrix::Matrix<std::complex<double> > aoR(input_file, aos, BH.Column, true, true);
    // (M, Nmu).
    DistributedMatrix::Matrix<std::complex<double> > aoR_mu(aoR.nrows, interp_indxs.size(), BH.Root);
    if (BH.rank == 0) std::cout << " * Down-sampling " << aos << " to find aoR_mu." << std::endl;
    MatrixOperations::down_sample_distributed_columns(aoR, aoR_mu, interp_indxs, BH);
    // aoR_mu's data is in fortran order. Need to transpose back for C order.
    if (BH.rank == 0) {
      MatrixOperations::local_transpose(aoR_mu, false);
      MatrixOperations::swap_dims(aoR_mu);
      if (write) {
        std::cout << " * Writing aoR_mu to file" << std::endl;
        H5::Exception::dontPrint();
        H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
        try {
          H5::Group base = file.createGroup("/Hamiltonian");
        } catch (H5::FileIException) {
          H5::Group base = file.openGroup("/Hamiltonian");
        }
        aoR_mu.dump_data(file, "/Hamiltonian/THC", prfx+"Orbitals");
        // Write interpolating indices
        std::vector<hsize_t> dims(1);
        dims[0] = interp_indxs.size();
        std::cout << " * Writing indices of interpolating points to file" << std::endl;
        H5Helper::write(file, "/Hamiltonian/THC/"+prfx+"InterpIndx", interp_indxs, dims);
      }
      // Transpose back to Fortran order.
      MatrixOperations::local_transpose(aoR_mu, true);
      MatrixOperations::swap_dims(aoR_mu);
    }
    // Finally construct pseudo density matrices matrices:
    //     [P]_{mu,g} = \sum_i \phi_i^{*}(r_mu) \phi_i(r_g),
    // where r_mu is an interpolating grid point and r_g is from the full real space grid.
    // [aoR]_{ig} = \phi_i(r_g) and [aoR_mu]_{imu} = \phi_i(r_\mu) are already in fortran
    // order. Need to Hermitian conjugate [aoR_mu].
    // Redistribute [aoR] block cyclically, in column major format.
    if (BH.rank == 0) std::cout << " * Redistributing " << aos << " block cyclically." << std::endl;
    MatrixOperations::redistribute(aoR, BH.Column, BH.Square, true, 64, 64);
    // [aoR_mu] is stored on the root processor with shape (M, N_mu).
    // Redistribute block cyclically.
    if (BH.rank == 0) std::cout << " * Redistributing sub-sampled " << aos << " block cyclically." << std::endl;
    MatrixOperations::redistribute(aoR_mu, BH.Root, BH.Square, true);
    // Now Hermitian transpose [aoR_mu].
    if (BH.rank == 0) std::cout << " * Transposing sub-sampled orbitals." << std::endl;
    Pua.setup_matrix(aoR_mu.ncols, aoR.ncols, BH.Square);
    if (BH.rank == 0) {
      std::cout << " * Constructing CZt matrix" << std::endl;
      std::cout << " * Matrix Shape: (" << CZt.nrows << ", " << CZt.ncols << ")" << std::endl;
      double memory = UTILS::get_memory(CZt.store);
      std::cout << "  * Local memory usage (on root processor): " << memory << " GB" << std::endl;
      std::cout << "  * Local shape (on root processor): (" << CZt.local_nrows << ", " << CZt.local_ncols << ")" << std::endl;
    }
    MatrixOperations::product(aoR_mu, aoR, Pua, 'C', 'N');
#ifndef NDEBUG
    // Print out pseudo density matrices.
    MatrixOperations::redistribute(Pua, BH.Square, BH.Root, true);
    Logging::dump_matrix(Pua, output_file, prfx+"Pua", false, BH.rank==0);
    // Transpose back to Fortran order.
    MatrixOperations::redistribute(Pua, BH.Root, BH.Square, true, 64, 64);
#endif
  }

  void IVecs::setup_CZt(std::vector<int> &interp_indxs, ContextHandler::BlacsHandler &BH)
  {
    // CZt = (aoR_mu^{*T} aoR)*(aoR_mu^{*T} aoR)^*,
    //     = P_{mu r} P_{mu r}^{*}
    // Where P are the orbital product matrices.
    setup_pseudo_dm(CZt, interp_indxs, BH, "aoR", true);
    for (int i = 0; i < CZt.store.size(); i++) {
      CZt.store[i] = CZt.store[i]*std::conj(CZt.store[i]);
    }
#ifndef NDEBUG
    MatrixOperations::redistribute(CZt, BH.Square, BH.Root);
    Logging::dump_matrix(CZt, output_file, prefix+"CZt", false, BH.rank==0);
    MatrixOperations::redistribute(CZt, BH.Root, BH.Square, false, 64, 64);
#endif
  }

  void IVecs::setup_CZt_half(std::vector<int> &interp_indxs, ContextHandler::BlacsHandler &BH)
  {
    // For the half transformed case we have:
    // CZt = (aoR_mu_occ^{*T} aoR_occ)*(aoR_mu^{*T} aoR)^*,
    //     = Pocc_{mu r} P_{mu r}^{*}
    // Where P_{mu r} = \sum_i (\phi_i^mu)*\phi_i^r and
    // Pocc_{mu r} \sum_a (\phi_a^mu)*\phi_a^r
    setup_pseudo_dm(CZt, interp_indxs, BH, "aoR", true, "HalfTransformedFull");
    DistributedMatrix::Matrix<std::complex<double> > Pua;
    setup_pseudo_dm(Pua, interp_indxs, BH, "aoR_half", true, "HalfTransformedOcc");
    for (int i = 0; i < CZt.store.size(); i++) {
      CZt.store[i] = Pua.store[i] * std::conj(CZt.store[i]);
    }
#ifndef NDEBUG
    MatrixOperations::redistribute(CZt, BH.Square, BH.Root);
    Logging::dump_matrix(CZt, output_file, prefix+"CZt", false, BH.rank==0);
    MatrixOperations::redistribute(CZt, BH.Root, BH.Square, false, 64, 64);
#endif
  }

  void IVecs::setup_CCt(std::vector<int> &interp_indxs, ContextHandler::BlacsHandler &BH)
  {
    // Need to select columns of CZt to construct CCt.
    // First redistribute CZt data column cyclically to avoid figuring out how to index
    // columns in scalapack.
    if (BH.rank == 0) {
      std::cout << " * Redistributing CZt column cyclically." << std::endl;
    }
    // Number of columns of CZt each processor will get i.e., [rank*ncols_per_block,(rank+1)*ncols_per_block]
    int ncols_per_block = ceil((double)CZt.ncols/BH.nprocs); // Check this, I think scalapack will take ceiling rather than floor.
    MatrixOperations::redistribute(CZt, BH.Square, BH.Column, true,
                                   CZt.nrows, ncols_per_block);
    CCt.setup_matrix(CZt.nrows, interp_indxs.size(), BH.Root);
    MatrixOperations::down_sample_distributed_columns(CZt, CCt, interp_indxs, BH);
    // Back to block cyclic distribution for linear algebra.
    if (BH.rank == 0) {
      std::cout << " * Redistributing CZt block cyclically." << std::endl;
    }
    MatrixOperations::redistribute(CZt, BH.Column, BH.Square, true, 64, 64);
#ifndef NDEBUG
    // Print out CCt matrices.
    Logging::dump_matrix(CCt, output_file, prefix+"CCt", false, BH.rank==0);
#endif
    if (BH.rank == 0) {
      std::cout << " * Redistributing CCt block cyclically." << std::endl;
    }
    MatrixOperations::redistribute(CCt, BH.Root, BH.Square, true, 64, 64);
  }

  void IVecs::check_rank(ContextHandler::BlacsHandler &BH)
  {
    std::vector<std::complex<double> > tmp(CCt.store);
    if (BH.rank == 0) std::cout << " * Checking rank of CCt matrix." << std::endl;
    int rank = MatrixOperations::rank(CCt, BH.Square);
    if (BH.rank == 0) {
      if (rank < CCt.nrows) {
        std::cout << " * WARNING: CCt matrix is rank deficient. Rank = " << rank << std::endl;
        std::cout << std::endl;
      } else {
        std::cout << " * CCt matrix has full rank : " << rank << std::endl;
        std::cout << std::endl;
      }
    }
    // SVD destroys data in CCt so copy it back.
    std::copy(tmp.begin(), tmp.end(), CCt.store.data());
  }

  void IVecs::dump_qmcpack_data(int thc_cfac, int thc_half_cfac,
                                ContextHandler::BlacsHandler &BH)
  {
    H5::Exception::dontPrint();
    H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
    H5::Group base = file.openGroup("/Hamiltonian");
    H5::H5File fin = H5::H5File(input_file.c_str(), H5F_ACC_RDONLY);
    // Read in and dump core hamiltonian
    std::vector<std::complex<double> > hcore;
    std::vector<hsize_t> dims(2);
    H5Helper::read_matrix(fin, "hcore", hcore, dims);
    H5Helper::write(file, "/Hamiltonian/hcore", hcore, dims);
    // Number of electrons (nup, ndown).
    std::vector<int> num_elec(2);
    H5Helper::read_matrix(fin, "num_electrons", num_elec, dims);
    int nup = num_elec[0], ndown = num_elec[1];
    // Constant energy factors (nuclear + Madelung).
    std::vector<double> energy_constants(2);
    H5Helper::read_matrix(fin, "constant_energy_factors", energy_constants, dims);
    dims.resize(1);
    dims[0] = 1;
    H5Helper::write(file, "/Hamiltonian/Energies", energy_constants, dims);
    // Other QMCPACK specific metadata.
    std::vector<int> occups(nup+ndown);
    MatrixOperations::zero(occups);
    dims[0] = occups.size();
    H5Helper::write(file, "/Hamiltonian/occups", occups, dims);
    std::vector<int> qmcpack_dims = {-1, 0, 0, nbasis, nup, ndown, 0, 0};
    dims[0] = qmcpack_dims.size();
    H5Helper::write(file, "/Hamiltonian/dims", qmcpack_dims, dims);
    int nmu = thc_cfac * nbasis;
    int nmu_half = thc_half_cfac * nbasis;
    std::vector<int> thc_dims = {nbasis, nmu, nmu_half};
    dims[0] = 3;
    H5Helper::write(file, "/Hamiltonian/THC/dims", thc_dims, dims);
  }

  void IVecs::construct_muv(DistributedMatrix::Matrix<std::complex<double> > &IVMG,
                            DistributedMatrix::Matrix<std::complex<double> > &Muv,
                            ContextHandler::BlacsHandler &BH)
  {
    // Read FFT of coulomb kernel.
    DistributedMatrix::Matrix<double> coulG(input_file, "fft_coulomb", BH.Root);
    // Resize on other processors.
    if (BH.rank != 0) coulG.store.resize(coulG.nrows*coulG.ncols);
    // Distribute FFT of coulomb kernel to all processors.
    MPI_Bcast(coulG.store.data(), coulG.store.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Recalling IVG data is stored in Fortran order.
    for (int j = 0; j < CZt.local_ncols; j++) {
      for (int i = 0; i < CZt.local_nrows; i++) {
        CZt.store[i+j*CZt.local_nrows] *= sqrt(coulG.store[i]);
        if (half_rotate) IVMG.store[i+j*IVMG.local_nrows] *= sqrt(coulG.store[i]);
      }
    }
    // Redistributed to block cyclic.
    if (BH.rank == 0) std::cout << " * Redistributing reciprocal space interpolating vectors to block cyclic distribution." << std::endl;
    MatrixOperations::redistribute(CZt, BH.Column, BH.Square, true, 64, 64);
    if (half_rotate) MatrixOperations::redistribute(IVMG, BH.Column, BH.Square, true, 64, 64);
    // (nmu, ngrid)
    if (BH.rank == 0) std::cout << " * Constructing Muv." << std::endl;
    double t_muv = clock();
    if (half_rotate) {
      // Computes M_{uv} = \sum_G \zeta_mu(-G) \zeta_nu(G).
      MatrixOperations::product(IVMG, CZt, Muv, 'T', 'N');
    } else {
      // Computes M_{uv} = \sum_G \zeta^{*}_mu(G) \zeta_nu(G).
      // Uses rank-k update routine which returns only lower ('L') part of Muv which is
      // consistent with Cholesky decomposition used later.
      MatrixOperations::product(CZt, Muv, 'C', 'L');
    }
    if (BH.rank == 0) std::cout << "  * Time to construct Muv: " << (clock()-t_muv) / CLOCKS_PER_SEC << " seconds." << std::endl;
  }

  void IVecs::dump_thc_data(DistributedMatrix::Matrix<std::complex<double> > &Muv,
                            ContextHandler::BlacsHandler &BH)
  {
    if (!half_rotate) {
      if (BH.rank == 0) std::cout << " * Performing Cholesky decomposition on Muv." << std::endl;
      double t_chol = clock();
      int ierr = MatrixOperations::cholesky(Muv);
      MatrixOperations::redistribute(Muv, BH.Square, BH.Root);
      if (BH.rank == 0) {
        H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
        H5::Group base = file.openGroup("/Hamiltonian");
        std::cout << " * Time to perform cholesky decomposition on Muv: " << (clock()-t_chol) / CLOCKS_PER_SEC << " seconds." << std::endl;
        std::cout << " * Dumping Luv to: " << output_file << "." << std::endl;
        if (ierr != 0) {
          std::cout << "  * WARNING: Cholesky decomposition failed!" << std::endl;
          std::cout << "  * SCALAPACK Error code: " << ierr << std::endl;
          if (ierr > 0) {
            std::cout << "   * The leading minor of order " << ierr << " is not positive definite." << std::endl;
          } else {
            std::cout << "   * Illegal value in matrix." << std::endl;
          }
        }
        std::cout << std::endl;
        // Zero out upper triangular bit of Luv which contains upper triangular part of Muv.
        for (int i = 0; i < Muv.nrows; i++) {
          for (int j = (i+1); j < Muv.ncols; j++) {
            // Luv is in fortran order.
            Muv.store[i+j*Muv.nrows] = 0.0;
          }
        }
        // Transform back to C order.
        MatrixOperations::local_transpose(Muv, false);
        Muv.dump_data(file, "/Hamiltonian/THC", prefix+"Luv");
      }
    } else {
      MatrixOperations::redistribute(Muv, BH.Square, BH.Root);
      if (BH.rank == 0) {
        H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
        H5::Group base = file.openGroup("/Hamiltonian");
        std::cout << " * Dumping half transformed Muv data to: " << output_file << "." << std::endl;
        std::cout << std::endl;
        // Transform back to C order.
        MatrixOperations::local_transpose(Muv, false);
        Muv.dump_data(file, "/Hamiltonian/THC", prefix+"Muv");
      }
    }
  }

  void IVecs::fft_vectors(ContextHandler::BlacsHandler &BH,
                          DistributedMatrix::Matrix<std::complex<double> > &IVMG)
  {
    // Need to transform interpolating vectors to C order so as to use FFTW and exploit
    // parallelism.  We actually solved for Theta^{*T}, so conjugate to get actual
    // interpolating vectors.
#ifndef NDEBUG
    MatrixOperations::redistribute(CZt, BH.Square, BH.Root);
    Logging::dump_matrix(CZt, output_file, prefix+"IVecs", false, BH.rank==0);
    MatrixOperations::redistribute(CZt, BH.Root, BH.Square, true, 64, 64);
#endif
    bool hermi = true;
    MatrixOperations::transpose(CZt, BH.Square, hermi);
    //MatrixOperations::swap_dims(CZt);
#ifndef NDEBUG
    // Print out Hermitian conjugate of interpolating vectors.
    MatrixOperations::redistribute(CZt, BH.Square, BH.Root);
    Logging::dump_matrix(CZt, output_file, prefix+"IVecs_conj", false, BH.rank==0);
    // Transpose back to Fortran order.
    MatrixOperations::redistribute(CZt, BH.Root, BH.Square);
#endif
    if (BH.rank == 0) {
      std::cout << " * Column cyclic CZt matrix info:" << std::endl;
    }
    // The `1' as the final argument cyclicially distributes the columns since we don't
    // care about the order.
    MatrixOperations::redistribute(CZt, BH.Square, BH.Column, true, CZt.nrows, 1);
    if (BH.rank == 0) {
      std::cout << std::endl;
    }
    // Finally we can FFT interpolating vectors
    fftw_plan p;
    int ngs = CZt.nrows;
    std::vector<std::complex<double> > tmp(ngs);
    if (BH.rank == 0) {
      std::cout << " * Performing FFT on grid with " << 2*fft_grid[0]+1 << " X " << 2*fft_grid[1]+1 << " X " << 2*fft_grid[2]+1 << " points." << std::endl;
    }
    if ((2*fft_grid[0]+1)*(2*fft_grid[1]+1)*(2*fft_grid[2]+1) != ngs) {
      if (BH.rank == 0) std::cout << " * WARNING: FFT grid not consitent with number of real space grid points." << std::endl;
    }
    for (int i = 0; i < CZt.local_ncols; i++) {
      // Data needs to be complex.
      std::copy(CZt.store.data()+i*ngs, CZt.store.data()+(i+1)*ngs,
                tmp.data());
      if (BH.rank == 0) {
        if ((i+1) % 20 == 0) std::cout << " * Performing FFT " << i+1 << " of " <<  CZt.local_ncols << std::endl;
      }
      p = fftw_plan_dft_3d(2*fft_grid[0]+1, 2*fft_grid[1]+1, 2*fft_grid[2]+1,
                           reinterpret_cast<fftw_complex*>(tmp.data()),
                           reinterpret_cast<fftw_complex*>(CZt.store.data()+i*ngs),
                           FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(p);
      fftw_destroy_plan(p);
      if (half_rotate) {
        p = fftw_plan_dft_3d(2*fft_grid[0]+1, 2*fft_grid[1]+1, 2*fft_grid[2]+1,
                             reinterpret_cast<fftw_complex*>(tmp.data()),
                             reinterpret_cast<fftw_complex*>(IVMG.store.data()+i*ngs),
                             FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
      }
    }
#ifndef NDEBUG
    // Print out FFTd interpolating vectors matrices.
    MatrixOperations::redistribute(CZt, BH.Column, BH.Root);
    Logging::dump_matrix(CZt, output_file, prefix+"IVG", false, BH.rank==0);
    MatrixOperations::redistribute(CZt, BH.Root, BH.Column);
    if (half_rotate) {
      MatrixOperations::redistribute(IVMG, BH.Column, BH.Root);
      Logging::dump_matrix(CZt, output_file, prefix+"IVMG", false, BH.rank==0);
      MatrixOperations::redistribute(IVMG, BH.Root, BH.Column);
    }
#endif
  }

  void IVecs::kernel(ContextHandler::BlacsHandler &BH)
  {
    if (BH.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << "##        Finding interpolating vectors.       ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      std::cout << " * Performing least squares solve." << std::endl;
    }
    double tlsq = clock();
    int ierr = MatrixOperations::least_squares(CCt, CZt);
    tlsq = clock() - tlsq;
    if (BH.rank == 0) {
      std::cout << " * Time for least squares solve : " << tlsq / CLOCKS_PER_SEC << " seconds" << std::endl;
      std::cout << std::endl;
      if (ierr != 0) {
        std::cout << " * Parallel least squares failed." << std::endl;
        std::cout << " * Error code: " << ierr << std::endl;
      }
    }

    if (BH.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << "##        Constructing Muv matrix.             ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      std::cout << " * Performing FFT on interpolating vectors." << std::endl;
      std::cout << std::endl;
    }
    int nrows, ncols;
    if (half_rotate) {
      nrows = CZt.ncols;
      ncols = CZt.nrows;
    } else {
      // Don't allocate the extra memory
      nrows = 1;
      ncols = 1;
    }
    double tfft = clock();
    DistributedMatrix::Matrix<std::complex<double> > IVMG(nrows, ncols, BH.Column, CZt.ncols, 1);
    fft_vectors(BH, IVMG);
    tfft = clock() - tfft;
    if (BH.rank == 0) {
      std::cout << " * Time to FFT interpolating vectors : " << tfft / CLOCKS_PER_SEC << " seconds" << std::endl;
      std::cout << std::endl;
    }
    DistributedMatrix::Matrix<std::complex<double> > Muv(CZt.ncols, CZt.ncols, BH.Square);
    construct_muv(IVMG, Muv, BH);
    dump_thc_data(Muv, BH);
  }
}
