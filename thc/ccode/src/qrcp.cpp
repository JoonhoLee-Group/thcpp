#include <iostream>
#include <mpi.h>
#include <time.h>
#include <complex>
#include <stdlib.h>
#include "json.hpp"
#include "h5helper.h"
#include "scalapack_defs.h"
#include "context_handler.h"
#include "matrix_operations.h"
#include "utils.h"
#include "logging.h"
#include "qrcp.h"

namespace QRCP
{
  // Constructor.
  QRCPSolver::QRCPSolver(nlohmann::json &input, int cfac, ContextHandler::BlacsHandler &BH)
  {
    if (BH.rank == 0) {
      input_file = input.at("orbital_file").get<std::string>();
      filename_size = input_file.size();
    }
    thc_cfac = cfac;
    MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&thc_cfac, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (BH.rank != 0) input_file.resize(filename_size);
    MPI_Bcast(&input_file[0], input_file.size()+1, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  // Main driver to find interpolating vectors via QRCP solve.
  void QRCPSolver::kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs)
  {
    if (BH.rank == 0) {
      std::cout << " * Finding interpolating points via QRCP." << std::endl;
    }
    // Will read into [M,N_G] matrix in FTN format.
    // just explicitly construct Z for now and reproduce scipy.
    // (M, Ngrid).
    DistributedMatrix::Matrix<std::complex<double> > aoR(input_file, "aoR",
                                                         BH.Column, true, true);
    int nbasis = aoR.nrows;
    int M2 = nbasis * nbasis;
    int ncols_per_block = ceil((double)aoR.ncols/BH.nprocs);
    DistributedMatrix::Matrix<std::complex<double> > ZT(M2, aoR.ncols, BH.Column,
                                                        M2, ncols_per_block);
    // Construct ZT matrix
    // [ZT]_{(ik)a} = \phi_i^*(r_a) \phi_k(r_a).
    // [todo]: replace with dger eventually.
    // Recall, ZT is stored in (fortran) column major order at this point.
    if (BH.rank == 0) {
      std::cout << " * Constructing Z matrix." << std::endl;
    }
    double tzmat = clock();
    for (int a = 0; a < ZT.local_ncols; a++) {
      for (int i = 0; i < nbasis; i++) {
        for (int k = 0; k < nbasis; k++) {
          ZT.store[a*M2+i*nbasis+k] = std::conj(aoR.store[a*nbasis+i]) * aoR.store[a*nbasis+k];
        }
      }
    }
    tzmat = clock() - tzmat;
    if (BH.rank == 0) {
      std::cout << "  * Time to construct Z matrix: " << tzmat / CLOCKS_PER_SEC << " seconds." << std::endl;
      std::cout << " * Redistributing ZT block cyclically." << std::endl;
    }
      //for (int i = 0; i < nbasis; i++) {
        //for (int k = 0; k < nbasis; k++) {
          //for (int a = 0; a < ZT.local_ncols; a++) {
            //std::cout << std::setprecision(16) << ZT.store[a*M2+i*nbasis+k].real() << " ";
          //}
          //std::cout << "XXX" << std::endl;
        //}
      //}
    //DistributedMatrix::Matrix<std::complex<double> > ZZT(M2, aoR.ncols, BH.Column,
                                                         //M2, ncols_per_block);
    //std::copy(ZT.store.begin(), ZT.store.end(), ZZT.store.begin());
    MatrixOperations::redistribute(ZT, BH.Column, BH.Square, true, 64, 64);
    //MatrixOperations::redistribute(ZZT, BH.Column, BH.Square, true, 64, 64);
    std::vector<int> perm;
    if (BH.rank == 0) {
      std::cout << " * Performing QRCP solve." << std::endl;
    }
    //int rank = MatrixOperations::rank(ZZT, BH.Square, true);
    MatrixOperations::qrcp(ZT, perm, BH.Square);
    int num_interp_pts = thc_cfac * nbasis;
    interp_indxs.resize(num_interp_pts);
    // Work out diagonal entries.
    MatrixOperations::redistribute(ZT, BH.Square, BH.Column, true,
                                   M2, ncols_per_block);
    std::vector<double> diag(ZT.local_ncols), global_diag(ZT.ncols);
    int offset = BH.rank * ncols_per_block;
    int max_diag = std::min(ZT.nrows, ZT.ncols);
    int ndiag_per_proc = max_diag - offset;
    if (ndiag_per_proc >= 0) {
      for (int i = 0; i < ZT.local_ncols; i++) {
        diag[i] = std::abs(ZT.store[i*M2+i+offset].real());
      }
    }
    std::vector<int> recv_counts(BH.nprocs), disps(BH.nprocs);
    int num_cols;
    MPI_Gather(&ZT.local_ncols, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    disps[0] = 0;
    for (int i = 1; i < recv_counts.size(); i++) {
      disps[i] = disps[i-1] + recv_counts[i-1];
    }
    MPI_Gatherv(diag.data(), diag.size(), MPI_DOUBLE,
                global_diag.data(), recv_counts.data(), disps.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    if (BH.rank == 0) {
      std::copy(perm.begin(), perm.begin()+num_interp_pts, interp_indxs.data());
      std::sort(interp_indxs.begin(), interp_indxs.end());
      int offset = ZT.nrows;
      for (int i = 0; i < max_diag; i++) {
        std::cout << "IX: " << i << " " << perm[i] << " " << std::setprecision(16) << global_diag[i] << std::endl;
      }
    }
  }

  // Destructor.
  QRCPSolver::~QRCPSolver()
  {
  }
}
