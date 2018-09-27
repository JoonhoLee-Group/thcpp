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
  QRCPSolver::QRCPSolver(nlohmann::json &input, ContextHandler::BlacsHandler &BH)
  {
    if (BH.rank == 0) {
      input_file = input.at("orbital_file").get<std::string>();
      try {
        sub_sample = input.at("interpolating_points").at("qrcp").at("sub_sample").get<bool>();
      }
      catch (nlohmann::json::out_of_range& error) {
        std::cout << " * Performing full QRCP decomposition." << std::endl;
      }
      filename_size = input_file.size();
    }
    MPI_Bcast(&filename_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (BH.rank != 0) input_file.resize(filename_size);
    MPI_Bcast(&input_file[0], input_file.size()+1, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  void setup_Z_matrix(DistributedMatrix::Matrix<std::complex<double> > &ZT,
                      DistributedMatrix::Matrix<std::complex<double> > &aoR)
  {
    // Construct ZT matrix
    // [ZT]_{(ik)a} = \phi_i^*(r_a) \phi_k(r_a).
    // [todo]: replace with dger eventually.
    // Recall, ZT is stored in (fortran) column major order at this point.
    int nbasis = aoR.nrows;
    int M2 = ZT.nrows;
    for (int a = 0; a < ZT.local_ncols; a++) {
      for (int i = 0; i < aoR.nrows; i++) {
        for (int k = 0; k < aoR.nrows; k++) {
          ZT.store[a*M2+i*nbasis+k] = std::conj(aoR.store[a*nbasis+i]) * aoR.store[a*nbasis+k];
        }
      }
    }
  }
  void setup_Z_half_matrix(DistributedMatrix::Matrix<std::complex<double> > &ZT,
                           DistributedMatrix::Matrix<std::complex<double> > &aoR,
                           DistributedMatrix::Matrix<std::complex<double> > &aoR_half)
  {
    // Construct ZT matrix
    // [ZT]_{(ik)a} = \phi_i^*(r_a) \phi_k(r_a).
    // [todo]: replace with dger eventually.
    // Recall, ZT is stored in (fortran) column major order at this point.
    int nbasis = aoR.nrows;
    int nelec = aoR_half.nrows;
    int MN = ZT.nrows;
    for (int a = 0; a < ZT.local_ncols; a++) {
      for (int i = 0; i < aoR_half.nrows; i++) {
        for (int k = 0; k < aoR.nrows; k++) {
          ZT.store[a*MN+i*nelec+k] = std::conj(aoR_half.store[a*nelec+i]) * aoR.store[a*nbasis+k];
        }
      }
    }
  }

  std::vector<int> QRCPSolver::map_perm(std::vector<int> &perm, int ncols, int local_ncols, ContextHandler::BlacsHandler &BH)
  {
    std::vector<int> global_index, col_index(ncols), global_perm(ncols);
    int nz = 0;
    for (int i = 0; i < local_ncols; ++i) {
      if (perm[i] >= 0 && BH.Square.row == 0) {
        nz += 1;
        global_index.push_back(MatrixOperations::global_matrix_col_index(i, BH.Square, 64));
      }
    }
    std::vector<int> recv_counts(BH.nprocs), disps(BH.nprocs);
    int num_cols;
    MPI_Gather(&nz, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    disps[0] = 0;
    for (int i = 1; i < recv_counts.size(); i++) {
      disps[i] = disps[i-1] + recv_counts[i-1];
      if (BH.rank == 0) {
      }
    }
    MPI_Gatherv(global_index.data(), global_index.size(), MPI_INT,
                col_index.data(), recv_counts.data(), disps.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(perm.data(), nz, MPI_INT,
                global_perm.data(), recv_counts.data(), disps.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    if (BH.rank == 0) {
      UTILS::sort_a_from_b(global_perm, col_index);
    }
    return global_perm;
  }
  // Main driver to find interpolating vectors via QRCP solve.
  void QRCPSolver::kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs,
                          int thc_cfac, bool half_rotate)
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
    int ncols_per_block = ceil((double)aoR.ncols/BH.nprocs);
    DistributedMatrix::Matrix<std::complex<double> > ZT;
    if (BH.rank == 0) {
      std::cout << " * Constructing Z matrix." << std::endl;
    }
    if (half_rotate) {
      DistributedMatrix::Matrix<std::complex<double> > aoR_half(input_file, "aoR_half",
                                                                BH.Column, true, true);
      int nelec = aoR_half.nrows;
      int MN = nbasis * nelec;
      ZT.setup_matrix(MN, aoR.ncols, BH.Column, MN, ncols_per_block);
      setup_Z_half_matrix(ZT, aoR, aoR_half);
    } else {
      int M2 = nbasis * nbasis;
      ZT.setup_matrix(M2, aoR.ncols, BH.Column, M2, ncols_per_block);
      setup_Z_matrix(ZT, aoR);
    }
    double tzmat = clock();
    tzmat = clock() - tzmat;
    if (BH.rank == 0) {
      std::cout << "  * Time to construct Z matrix: " << tzmat / CLOCKS_PER_SEC << " seconds." << std::endl;
      std::cout << " * Redistributing ZT block cyclically." << std::endl;
    }
    //if (!half_rotate) {
        //int M2 = nbasis * nbasis;
        //for (int i = 0; i < nbasis; i++) {
          //for (int k = 0; k < nbasis; k++) {
            //for (int a = 0; a < ZT.local_ncols; a++) {
              //std::cout << std::setprecision(16) << ZT.store[a*M2+i*nbasis+k].real() << " ";
            //}
            //std::cout << "XXX" << std::endl;
          //}
        //}
    //}
    //DistributedMatrix::Matrix<std::complex<double> > ZZT(M2, aoR.ncols, BH.Column,
                                                         //M2, ncols_per_block);
    //std::copy(ZT.store.begin(), ZT.store.end(), ZZT.store.begin());
    MatrixOperations::redistribute(ZT, BH.Column, BH.Square, true, 64, 64);
    //MatrixOperations::redistribute(ZZT, BH.Column, BH.Square, true, 64, 64);
    std::vector<int> perm;
    //int rank = MatrixOperations::rank(ZZT, BH.Square, true);
    MatrixOperations::qrcp(ZT, perm, BH.Square);
    // Work out diagonal entries.
    //MatrixOperations::redistribute(ZT, BH.Square, BH.Column, true,
                                   //ZT.nrows, ncols_per_block);
    //int offset = BH.rank * ncols_per_block;
    //int max_diag = std::min(ZT.nrows, ZT.ncols);
    //int ndiag_per_proc = max_diag - offset;
    //std::vector<double> diag(local_ncols), global_diag(ZT.ncols);
    //if (ndiag_per_proc >= 0) {
      //for (int i = 0; i < max_diag; i++) {
        //diag[i] = std::abs(ZT.store[i*ZT.nrows+i+offset].real());
      //}
    //}
    //int rank = MatrixOperations::rank(ZZT, BH.Square, true);
    std::vector<int> ordered_perm = map_perm(perm, ZT.ncols, ZT.local_ncols, BH);
    int num_interp_pts = thc_cfac * nbasis;
    interp_indxs.resize(num_interp_pts);
    if (BH.rank == 0) {
      std::copy(ordered_perm.begin(), ordered_perm.begin()+num_interp_pts, interp_indxs.data());
      std::sort(interp_indxs.begin(), interp_indxs.end());
      //for (int i = 0; i < interp_indxs.size(); i++) {
        ////std::cout << i << " " << MatrixOperations::global_matrix_index(i, BH.Square, 64) << " " << perm[i] << std::endl;
        //std::cout << i << " " << interp_indxs[i] << std::endl;
      //}
      //int offset = ZT.nrows;
      //std::string prefix;
      //if (half_rotate) {
        //prefix = "half_ix";
      //} else {
        //prefix = "full_ix";
      //}
      //std::cout << max_diag << " " << perm.size() << " " << ZT.nrows << " " << ZT.ncols << std::endl;
      //for (int i = 0; i < max_diag; i++) {
        //std::cout << prefix << " " << i << " " << perm[i] << " " << std::setprecision(16) << global_diag[i] << std::endl;
      //}
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Destructor.
  QRCPSolver::~QRCPSolver()
  {
  }
}
