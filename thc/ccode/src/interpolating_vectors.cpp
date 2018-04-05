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
  IVecs::IVecs(nlohmann::json &input, ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs, DistributedMatrix::Matrix<double> &aoR)
  {
    input_file = input.at("orbital_file").get<std::string>();
    output_file = input.at("output_file").get<std::string>();
    // (Nmu, M)
    DistributedMatrix::Matrix<double> aoR_mu(interp_indxs.size(), aoR.ncols, BH.Root);
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
    }
    // First need to modify matrices to fortran format.
    // CZt = aoR_mu aoR^T,
    // aoR is in C order, so already transposed from Fortran's perspective, just need to
    // alter nrows and ncols.
    int tmp = aoR.nrows;
    aoR.nrows = aoR.ncols;
    aoR.ncols = tmp;
    // Done on single processor.
    aoR.initialise_descriptor(aoR.desc, BH.Root, aoR.local_nrows, aoR.local_ncols);
    // Actually do need to transpose aoR_mu's data.
    if (BH.rank == 0) {
      std::vector<double> tmp(aoR_mu.store.size());
      for (int i = 0; i < aoR_mu.nrows; ++i) {
        for (int j = 0; j < aoR_mu.ncols; ++j) {
          tmp[j*aoR_mu.nrows+i] = aoR_mu.store[i*aoR_mu.ncols+j];
        }
      }
      aoR_mu.store.swap(tmp);
    }
    // But shape (Nmu, M) should stay the same.
    // Finally construct CZt.
    if (BH.rank == 0) {
      std::cout << " * Constructing CZt matrix" << std::endl;
    }
    CZt.setup_matrix(aoR_mu.nrows, aoR.ncols, BH.Root);
    CCt.setup_matrix(CZt.nrows, interp_indxs.size(), BH.Root);
    if (BH.rank == 0) {
      MatrixOperations::product(aoR_mu, aoR, CZt);
      // Hadamard products.
      for (int i = 0; i < CZt.store.size(); ++i) {
        CZt.store[i] *= CZt.store[i];
      }
      //std::cout << MatrixOperations::vector_sum(CZt.store) << std::endl;
      std::cout << " * Matrix Shape: (" << CZt.nrows << ", " << CZt.ncols << ")" << std::endl;
      std::cout << " * Constructing CCt matrix" << std::endl;
      std::cout << " * Matrix Shape: (" << CCt.nrows << ", " << CCt.ncols << ")" << std::endl;
      // Need to down sample columns of CZt to form CCt, both of which have their data store
      // in fortran / column major order.
      MatrixOperations::down_sample(CZt, CCt, interp_indxs, CZt.nrows);
      //std::cout << MatrixOperations::vector_sum(CCt.store) << std::endl;
    }
    // Block cyclically distribute.
    if (BH.rank == 0) {
      std::cout << " * Block cyclic CZt matrix info:" << std::endl;
    }
    MatrixOperations::redistribute(CZt, BH.Root, BH.Square, true);
    if (BH.rank == 0) {
      std::cout << " * Block cyclic CCt matrix info:" << std::endl;
    }
    MatrixOperations::redistribute(CCt, BH.Root, BH.Square, true);
    if (BH.rank == 0) std::cout << std::endl;
  }

  void IVecs::dump_thc_data(DistributedMatrix::Matrix<std::complex<double> > &IVG, ContextHandler::BlacsHandler &BH)
  {
    // Read FFT of coulomb kernel.
    DistributedMatrix::Matrix<double> coulG(input_file, "fft_coulomb", BH.Root);
    // Resize on other processors.
    if (coulG.store.size() != coulG.nrows*coulG.ncols) coulG.store.resize(coulG.nrows*coulG.ncols);
    MPI_Bcast(coulG.store.data(), coulG.store.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Distribute FFT of coulomb kernel to all processors.
    //std::complex<double> local_sum = MatrixOperations::vector_sum(IVG.store), global_sum = 0.0;
    //MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    //if (BH.rank == 0) std::cout << "SUM of IVG: " << global_sum << std::endl;
    //MPI_Bcast(coulG.store.data(), coulG.store.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    // overload product for complex data type.
    if (BH.rank == 0) std::cout << " * Constructing Muv." << std::endl;
    MatrixOperations::product(IVGT, IVG, Muv);
    // Dump matrices to file.
    MatrixOperations::redistribute(Muv, BH.Square, BH.Root);
    if (BH.rank == 0) {
      std::cout << " * Dumping THC data to: " << output_file << "." << std::endl;
      std::cout << std::endl;
      H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
      H5::Group base = file.openGroup("/Hamiltonian");
      Muv.dump_data(file, "/Hamiltonian/THC", "Muv");
    }
  }

  void IVecs::fft_vectors(ContextHandler::BlacsHandler &BH, DistributedMatrix::Matrix<std::complex<double> > &IVG)
  {
    // Need to transform interpolating vectors to C order so as to use FFTW and exploit
    // parallelism.
    DistributedMatrix::Matrix<double> CZ(CZt.ncols, CZt.nrows, BH.Square);
    MatrixOperations::transpose(CZt, CZ);
    // will now have matrix in C order with local shape (nmu, ngs)
    //double local_sum = MatrixOperations::vector_sum(CZ.store);
    //double global_sum = 0.0;
    //MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (BH.rank == 0) {
      std::cout << " * Column cyclic CZ matrix info:" << std::endl;
    }
    // Fortran sees this as a (ngrid, nmu) matrix, so we can distributed vectors of length
    // ngrid cyclically to each processor.
    //local_sum = MatrixOperations::vector_sum(CZ.store);
    //global_sum = 0.0;
    //MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //if (BH.rank == 0) std::cout << "Before: " << MatrixOperations::vector_sum(CZ.store) << " " << global_sum << std::endl;
    MatrixOperations::redistribute(CZ, BH.Square, BH.Column, true, CZ.nrows, 1);
    //local_sum = MatrixOperations::vector_sum(CZ.store);
    //global_sum = 0.0;
    //MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (BH.rank == 0) {
      std::cout << std::endl;
      //std::cout << "AFter: " << MatrixOperations::vector_sum(CZ.store) << " " << global_sum << std::endl;
    }
    // Finally we can FFT interpolating vectors
    fftw_plan p;
    int ngs = CZ.nrows;
    int ng = (int)pow(ngs, 1.0/3.0);
    int offset = ngs;
    //std::cout << ng << " " << ngs << " " << offset <<  std::endl;
    for (int i = 0; i < CZ.local_ncols; i++) {
      // Data needs to be complex.
      std::vector<std::complex<double> > complex_data = UTILS::convert_double_to_complex(CZ.store.data()+i*offset, ngs);
      if (BH.rank == 0) {
        if ((i+1) % 20 == 0) std::cout << " * Performing FFT " << i+1 << " of " <<  CZ.local_ncols << std::endl;
      }
      // there is a routine for many FFTs run into integer overflow here.
      p = fftw_plan_dft_3d(ng, ng, ng, reinterpret_cast<fftw_complex*> (complex_data.data()),
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
      //std::cout << MatrixOperations::vector_sum(CZt.store) << std::endl;
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
