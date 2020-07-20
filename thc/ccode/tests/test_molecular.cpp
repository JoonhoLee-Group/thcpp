#include "catch.hpp"
#include <iostream>

#include "distributed_matrix.h"
#include "context_handler.h"
#include "matrix_operations.h"

TEST_CASE("test_mol_thc", "[molecular]")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> orbs("eri_thc.h5", "etaPm", BH.Root, true, false);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack
    MatrixOperations::local_transpose(orbs);
    MatrixOperations::swap_dims(orbs);
  }
  int nthc = orbs.nrows;
  int norb = orbs.ncols;
  MatrixOperations::redistribute(orbs, BH.Root, BH.Square, true);
  DistributedMatrix::Matrix<double> S(orbs.nrows, orbs.nrows, BH.Square), T2(orbs.nrows, orbs.nrows, BH.Square);
  MatrixOperations::product(orbs, orbs, S, 'N', 'T');
  MatrixOperations::hadamard_product(S);
  DistributedMatrix::Matrix<double> Sinv(orbs.nrows, orbs.nrows, BH.Square);
  MatrixOperations::pseudo_inverse(S, Sinv, 1e-12, BH);
  // Contract with df integrals
  DistributedMatrix::Matrix<double> df_ints("eri_thc.h5", "eri_df", BH.Root, true, false);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack
    MatrixOperations::local_transpose(df_ints);
    MatrixOperations::swap_dims(df_ints);
  }
  if (BH.rank == 0)
    std::cout << " * redistributing df ints." << std::endl;
  MatrixOperations::redistribute(df_ints, BH.Root, BH.Square, true);
  DistributedMatrix::Matrix<double> orbs_t(orbs.ncols, orbs.nrows, BH.Square);
  MatrixOperations::transpose(orbs, orbs_t);
  int msquare = orbs.ncols * orbs.ncols;
  int nbasis = orbs.ncols;
  int naux = df_ints.nrows;
  if (BH.rank == 0)
    std::cout << " * redistributing orbitals." << std::endl;
  // Form BxL
  MatrixOperations::redistribute(orbs_t, BH.Square, BH.Column, true, nbasis, 1);
  DistributedMatrix::Matrix<double> orb_prod(msquare, orbs.nrows, BH.Column, msquare, 1);
  REQUIRE(orb_prod.ncols == nthc);
  MatrixOperations::tensor_rank_one(orbs_t, orb_prod);
  MatrixOperations::redistribute(orb_prod, BH.Column, BH.Square);
  DistributedMatrix::Matrix<double> BxL(df_ints.nrows, orb_prod.ncols,  BH.Square);
  REQUIRE(BxL.nrows == naux);
  REQUIRE(BxL.ncols == nthc);
  MatrixOperations::redistribute(orb_prod, BH.Column, BH.Square);
  MatrixOperations::product(df_ints, orb_prod, BxL);
  // Form DxK
  DistributedMatrix::Matrix<double> DxK(naux, nthc);
  MatrixOperations::product(BxL, Sinv, DxK);
}
