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

namespace InterpolatingVectors
{
  IVecs::IVecs(nlohmann::json &input, ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs)
  {
    if (BH.rank == 0) {
      input_file = input.at("orbital_file").get<std::string>();
      output_file = input.at("output_file").get<std::string>();
      thc_cfac = input.at("thc_cfac").get<int>();
    }
    {
      // (Ngrid, M)
      DistributedMatrix::Matrix<std::complex<double> > aoR(input_file, "aoR", BH.Root);
      // (Nmu, M)
      DistributedMatrix::Matrix<std::complex<double> > aoR_mu(interp_indxs.size(), aoR.ncols, BH.Root);
      DistributedMatrix::Matrix<std::complex<double> > hcore(input_file, "hcore", BH.Root);
      if (BH.rank == 0) {
        std::cout << "#################################################" << std::endl;
        std::cout << "##   Setting up interpolative vector solver.   ##" << std::endl;
        std::cout << "#################################################" << std::endl;
        std::cout << std::endl;
        std::cout << " * Down-sampling aoR to find aoR_mu" << std::endl;
        std::cout << " * Writing aoR_mu to file" << std::endl;
        MatrixOperations::down_sample(aoR, aoR_mu, interp_indxs, aoR.ncols);
        H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_TRUNC);
        H5::Group base = file.createGroup("/Hamiltonian");
        aoR_mu.dump_data(file, "/Hamiltonian/THC", "orbitals");
        // Read in and dump core hamiltonian
        hcore.dump_data(file, "/Hamiltonian", "hcore");
        std::vector<hsize_t> dims(2);
        H5::H5File fin = H5::H5File(input_file.c_str(), H5F_ACC_RDONLY);
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
        int nbasis = aoR.ncols;
        std::vector<int> qmcpack_dims = {-1, 0, nbasis, nup, ndown, 0, 0, 0};
        dims[0] = qmcpack_dims.size();
        H5Helper::write(file, "/Hamiltonian/dims", qmcpack_dims, dims);
        int nmu = thc_cfac * nbasis;
        std::vector<int> thc_dims = {nbasis, nmu};
        dims[0] = 2;
        H5Helper::write(file, "/Hamiltonian/THC/dims", thc_dims, dims);
      }
      // First need to modify matrices to fortran format.
      // CZt = aoR_mu aoR^T,
      // aoR is in C order, so already transposed from Fortran's perspective, just need to
      // alter nrows and ncols.
      int tmp = aoR.nrows;
      aoR.nrows = aoR.ncols;
      aoR.ncols = tmp;
      aoR.initialise_descriptor(aoR.desc, BH.Root, aoR.local_nrows, aoR.local_ncols);
      // Actually do need to transpose aoR_mu's data.
      if (BH.rank == 0) {
        std::vector<std::complex<double> > tmp(aoR_mu.store.size());
        for (int i = 0; i < aoR_mu.nrows; ++i) {
          for (int j = 0; j < aoR_mu.ncols; ++j) {
            tmp[j*aoR_mu.nrows+i] = aoR_mu.store[i*aoR_mu.ncols+j];
          }
        }
        aoR_mu.store.swap(tmp);
      }
      // But shape (Nmu, M) should stay the same.
      // Finally construct CZt.
      MatrixOperations::redistribute(aoR, BH.Root, BH.Square, true);
      MatrixOperations::redistribute(aoR_mu, BH.Root, BH.Square, true);
      CZt.setup_matrix(aoR_mu.nrows, aoR.ncols, BH.Square);
      if (BH.rank == 0) {
        std::cout << " * Constructing CZt matrix" << std::endl;
        std::cout << " * Matrix Shape: (" << CZt.nrows << ", " << CZt.ncols << ")" << std::endl;
        double memory = UTILS::get_memory(CZt.store);
        std::cout << "  * Local memory usage (on root processor): " << memory << " GB" << std::endl;
        std::cout << "  * Local shape (on root processor): (" << CZt.local_nrows << ", " << CZt.local_ncols << ")" << std::endl;
      }
      MatrixOperations::product(aoR_mu, aoR, CZt);
    } // Memory from aoR and aoR_mu should be freed.
    // Hadamard products.
    for (int i = 0; i < CZt.store.size(); i++) {
      CZt.store[i] *= CZt.store[i];
    }
    // Need to select columns of CZt to construct CCt.
    // First redistribute CZt data column cyclically to avoid figuring out how to index
    // columns in scalapack.
    if (BH.rank == 0) {
      std::cout << " * Redistributing CZt column cyclically." << std::endl;
    }
    // Number of columns of CZt each processor will get i.e., [rank*ncols_per_block,(rank+1)*ncols_per_block]
    int ncols_per_block = CZt.ncols / BH.nprocs;
    MatrixOperations::redistribute(CZt, BH.Square, BH.Column, true, CZt.nrows, ncols_per_block);
    // Next figure out which columns to select on each processor.
    // We extend selected index array to be the same size as the CZt.ncols, so that we can
    // redistribute this array to align with CZt and don't need to figure out any indexing.
    // not sure if this is any easier than just working out the processor id of the index?
    DistributedMatrix::Matrix<double> ix_map(1, CZt.ncols, BH.Root);
    if (BH.rank == 0) {
      for (int i = 0; i < ix_map.store.size(); i++) {
          ix_map.store[i] = -1;
      }
      for (int i = 0; i < interp_indxs.size(); i++) {
        ix_map.store[interp_indxs[i]] = 1;
      }
    }
    // Redistribute to same processor grid as CZt.
    if (BH.rank == 0) {
      std::cout << " * Redistributing ix_map column cyclically." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MatrixOperations::redistribute(ix_map, BH.Root, BH.Column, true, 1, ncols_per_block);
    int num_cols = 0;
    {
      for (int i = 0; i < ix_map.store.size(); i++) {
        if (ix_map.store[i] > 0) {
          // Work out number of selected columns on current processor.
          // These will not be evenly distributed amongst processors.
          num_cols++;
        }
      }
      std::vector<std::complex<double> > local_cols(CZt.nrows*num_cols);
      num_cols = 0;
      // Second time around copy data.
      for (int i = 0; i < ix_map.store.size(); i++) {
        if (ix_map.store[i] > 0) {
          // Index in original global array. We need this to sort collected CCt later.
          // CZt is stored in Fortran format, so columns are contiguous in memory, which is
          // what we want.
          std::copy(CZt.store.begin()+i*CZt.nrows,
                    CZt.store.begin()+(i+1)*CZt.nrows,
                    local_cols.begin()+num_cols*CZt.nrows);
          num_cols++;
        }
      }
      // Work out how many columns of data we'll receive from each processor.
      std::vector<int> recv_counts(BH.nprocs), disps(BH.nprocs);
      // Figure out number of columns each processor will send to root.
      MPI_Gather(&num_cols, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
      disps[0] = 0;
      recv_counts[0] *= CZt.nrows;
      for (int i = 1; i < recv_counts.size(); i++) {
        disps[i] = disps[i-1] + recv_counts[i-1];
        recv_counts[i] *= CZt.nrows;
      }
      // Because interp_indxs is sorted and we've chunked CZt in an ordered way, then the
      // selected columns in local_cols will be places in CCt in such a way so as to match
      // the order in aoR_mu, the down sampled atomic orbitals at the interpolating points.
      CCt.setup_matrix(CZt.nrows, interp_indxs.size(), BH.Root);
      MPI_Gatherv(local_cols.data(), num_cols*CCt.nrows, MPI_DOUBLE_COMPLEX,
                  CCt.store.data(), recv_counts.data(), disps.data(), MPI_DOUBLE_COMPLEX,
                  0, MPI_COMM_WORLD);
    } // Memory from local stores should be freed.
    // Back to block cyclic distribution for linear algebra.
    if (BH.rank == 0) {
      std::cout << " * Redistributing CZt block cyclically." << std::endl;
    }
    MatrixOperations::redistribute(CZt, BH.Column, BH.Square, true, 64, 64);
    if (BH.rank == 0) {
      std::cout << " * Redistributing CCt block cyclically." << std::endl;
    }
    MatrixOperations::redistribute(CCt, BH.Root, BH.Square, true, 64, 64);
    if (BH.rank == 0) {
      std::cout << std::endl;
    }
  }

  void IVecs::dump_thc_data(DistributedMatrix::Matrix<std::complex<double> > &IVG, ContextHandler::BlacsHandler &BH)
  {
    // Read FFT of coulomb kernel.
    DistributedMatrix::Matrix<double> coulG(input_file, "fft_coulomb", BH.Root);
    // Resize on other processors.
    if (BH.rank != 0) coulG.store.resize(coulG.nrows*coulG.ncols);
    MPI_Bcast(coulG.store.data(), coulG.store.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Distribute FFT of coulomb kernel to all processors.
    std::complex<double> local_sum = MatrixOperations::vector_sum(IVG.store), global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    // Recall IVGs store is in fortran order so transposed for us.
    for (int i = 0; i < IVG.local_ncols; i++) {
      for (int j = 0; j < IVG.local_nrows; j++) {
        IVG.store[j+i*IVG.local_nrows] *= sqrt(coulG.store[j]);
      }
    }
    //std::complex<double> ls = MatrixOperations::vector_sum(IVG.store);
    //std::complex<double> gs = 0.0;
    //MPI_Reduce(&ls, &gs, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    //if (BH.rank == 0) std::cout << "Scaled COUL: " << gs << std::endl;
    // Redistributed to block cyclic.
    // magic numbers..
    if (BH.rank == 0) std::cout << " * Redistributing reciprocal space interpolating vectors to block cyclic distribution." << std::endl;
    MatrixOperations::redistribute(IVG, BH.Column, BH.Square, true, 64, 64);
    DistributedMatrix::Matrix<std::complex<double> > IVGT(IVG.ncols, IVG.nrows, BH.Square);
    DistributedMatrix::Matrix<std::complex<double> > Muv(IVG.ncols, IVG.ncols, BH.Square);
    // overload product for complex data type.
    MatrixOperations::transpose(IVG, IVGT);
    for (int i = 0; i < IVG.store.size(); i++) {
      IVG.store[i] = std::conj(IVG.store[i]);
    }
    // numpy.dot(IVG, IVG.conj().T)
    if (BH.rank == 0) std::cout << " * Constructing Muv." << std::endl;
    double t_muv = clock();
    MatrixOperations::product(IVGT, IVG, Muv);
    if (BH.rank == 0) std::cout << "  * Time to construct Muv: " << (clock()-t_muv) / CLOCKS_PER_SEC << " seconds." << std::endl;

    DistributedMatrix::Matrix<std::complex<double> > Luv = Muv;
    if (BH.rank == 0) std::cout << " * Performing Cholesky decomposition on Muv." << std::endl;
    double t_chol = clock();
    MatrixOperations::cholesky(Luv);
    if (BH.rank == 0) std::cout << "  * Time to perform cholesky decomposition on Muv: " << (clock()-t_chol) / CLOCKS_PER_SEC << " seconds." << std::endl;

    // Dump matrices to file.
    MatrixOperations::redistribute(Muv, BH.Square, BH.Root);
    MatrixOperations::redistribute(Luv, BH.Square, BH.Root);
    if (BH.rank == 0) {
      std::cout << " * Dumping THC data to: " << output_file << "." << std::endl;
      std::cout << std::endl;
      H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
      H5::Group base = file.openGroup("/Hamiltonian");
      // Actually Muv^T, Muv should be Hermitian..
      Muv.dump_data(file, "/Hamiltonian/THC", "Muv");
      // Zero out upper triangular bit of Luv which contains upper triangular part of Muv.
      for (int i = 0; i < Luv.nrows; i++) {
        for (int j = (i+1); j < Luv.ncols; j++) {
          // Luv is in fortran order.
          Luv.store[i+j*Luv.nrows] = 0.0;
        }
      }
      // Transform back to C order.
      DistributedMatrix::Matrix<std::complex<double> > LuvT(Luv.nrows, Luv.ncols, BH.Root);
      MatrixOperations::transpose(Luv, LuvT);
      LuvT.dump_data(file, "/Hamiltonian/THC", "Luv");
    }
  }

  void IVecs::fft_vectors(ContextHandler::BlacsHandler &BH, DistributedMatrix::Matrix<std::complex<double> > &IVG)
  {
    // Need to transform interpolating vectors to C order so as to use FFTW and exploit
    // parallelism.
    DistributedMatrix::Matrix<std::complex<double> > CZ(CZt.ncols, CZt.nrows, BH.Square);
    MatrixOperations::transpose(CZt, CZ);
    // will now have matrix in C order with local shape (nmu, ngs)
    if (BH.rank == 0) {
      std::cout << " * Column cyclic CZ matrix info:" << std::endl;
    }
    // Fortran sees this as a (ngrid, nmu) matrix, so we can distributed vectors of length
    // ngrid cyclically to each processor.
    MatrixOperations::redistribute(CZ, BH.Square, BH.Column, true, CZ.nrows, 1);
    if (BH.rank == 0) {
      std::cout << std::endl;
    }
    // Finally we can FFT interpolating vectors
    fftw_plan p;
    int ngs = CZ.nrows;
    int ng = (int)pow(ngs, 1.0/3.0);
    int offset = ngs;
    for (int i = 0; i < CZ.local_ncols; i++) {
      // Data needs to be complex.
      //std::vector<std::complex<double> > complex_data = UTILS::convert_double_to_complex(CZ.store.data()+i*offset, ngs);
      if (BH.rank == 0) {
        if ((i+1) % 20 == 0) std::cout << " * Performing FFT " << i+1 << " of " <<  CZ.local_ncols << std::endl;
      }
      // there is a routine for many FFTs run into integer overflow here.
      p = fftw_plan_dft_3d(ng, ng, ng,
                           reinterpret_cast<fftw_complex*> (CZ.store.data()+i*offset),
                           reinterpret_cast<fftw_complex*>(IVG.store.data()+i*offset),
                           FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(p);
      fftw_destroy_plan(p);
    }
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
    MatrixOperations::least_squares(CCt, CZt);
    tlsq = clock() - tlsq;
    if (BH.rank == 0) {
      std::cout << " * Time for least squares solve : " << tlsq / CLOCKS_PER_SEC << " seconds" << std::endl;
      std::cout << std::endl;
    }

    if (BH.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << "##        Constructing Muv matrix.             ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      std::cout << " * Performing FFT on interpolating vectors." << std::endl;
      std::cout << std::endl;
    }
    DistributedMatrix::Matrix<std::complex<double> > IVG(CZt.ncols, CZt.nrows, BH.Column, CZt.ncols, 1);
    fft_vectors(BH, IVG);
    if (BH.rank == 0) std::cout << std::endl;
    dump_thc_data(IVG, BH);
  }
}
