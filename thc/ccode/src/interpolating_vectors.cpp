#include <iostream>
#include <mpi.h>
#include <time.h>
#include "interpolating_vectors.h"
#include "h5helper.h"
#include "scalapack_defs.h"
#include "context_handler.h"
#include "matrix_operations.h"

namespace InterpolatingVectors
{
  IVecs::IVecs(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs, DistributedMatrix::Matrix &aoR)
  {
    // (Nmu, M)
    DistributedMatrix::Matrix aoR_mu(interp_indxs.size(), aoR.ncols, BH.Root);
    if (BH.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << "##   Setting up interpolative vector solver.   ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      MatrixOperations::down_sample(aoR, aoR_mu, interp_indxs, aoR.ncols);
      std::cout << " * Down-sampling aoR to find aoR_mu" << std::endl;
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    // First need to modify matrices to fortran format.
    // CZt = aoR_mu aoR^T,
    // aoR is in C order, so already transposed from Fortran's perspective, just need to
    // alter nrows and ncols.
    int tmp = aoR.nrows;
    aoR.nrows = aoR.ncols;
    aoR.ncols = tmp;
    // Done on single processor.
    aoR.initialise_discriptor(aoR.desc, BH.Root, aoR.local_nrows, aoR.local_ncols);
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
    //MPI_Barrier(MPI_COMM_WORLD);
    // But shape (Nmu, M) should stay the same.
    //tmp = aoR_mu.nrows;
    //aoR_mu.nrows = aoR_mu.ncols;
    //aoR_mu.ncols = tmp;
    //aoR_mu.initialise_discriptor(aoR_mu_T.desc, BH.Root, aoR_mu_T.local_ncols, aoR_mu_T.local_nrows);
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
    CZt.redistribute(BH.Root, BH.Square, true);
    if (BH.rank == 0) {
      std::cout << " * Block cyclic CCt matrix info:" << std::endl;
    }
    CCt.redistribute(BH.Root, BH.Square, true);
    if (BH.rank == 0) std::cout << std::endl;
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
    CZt.redistribute(BH.Square, BH.Root);
    tlsq = clock() - tlsq;
    if (BH.rank == 0) {
      //std::cout << MatrixOperations::vector_sum(CZt.store) << std::endl;
      std::cout << " * Time for least squares solve : " << tlsq / CLOCKS_PER_SEC << " seconds" << std::endl;
      std::cout << std::endl;
    }
  }
}
