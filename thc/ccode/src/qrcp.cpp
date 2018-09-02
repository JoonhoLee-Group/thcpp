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
      filename = input.at("orbital_file").get<std::string>();
    }
    thc_cfac = cfac;
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
    DistributedMatrix::Matrix<std::complex<double> > aoR(filename, "aoR",
                                                         BH.Column, true, true);
    int nbasis = aoR.nrows;
    int M2 = nbasis * nbasis;
    num_interp_pts = thc_cfac * nbasis;
    DistributedMatrix::Matrix<std::complex<double> > ZT(M2, aoR.ncols, BH.Root);
    // Construct ZT matrix
    // [ZT]_{(ik)a} = \phi_i^*(r_a) \phi_k(r_a).
    // [todo]: replace with dger eventually.
    // Recall, ZT is stored in (fortran) column major order at this point.
    if (BH.rank == 0) {
      std::cout << " * Constructing Z matrix." << std::endl;
    }
    for (int a = 0; a < ZT.local_ncols; a++) {
      for (int i = 0; i < nbasis; i++) {
        for (int k = 0; k < nbasis; k++) {
          ZT.store[a*M2+i*nbasis+k] = std::conj(aoR.store[a*nbasis+i]) * aoR.store[a*nbasis+k];
        }
      }
    }
    //MatrixOperations::redistribute(ZT, BH.Column, BH.Root);
    std::vector<int> perm;
    if (BH.rank == 0) {
      std::cout << " * Performing QRCP solve." << std::endl;
    }
    MatrixOperations::qrcp(ZT, perm, BH.Root);
    interp_indxs.resize(num_interp_pts);
    std::copy(perm.begin(), perm.begin()+num_interp_pts, interp_indxs.data());
    std::sort(interp_indxs.begin(), interp_indxs.end());
#ifndef NDEBUG
    for (int i = 0; i < interp_indxs.size(); i++) {
      std::cout << interp_indxs[i] << std::endl;
    }
#endif
  }

  // Destructor.
  QRCPSolver::~QRCPSolver()
  {
  }
}
