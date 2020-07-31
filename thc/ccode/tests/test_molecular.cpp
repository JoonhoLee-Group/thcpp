#include "catch.hpp"
#include <iostream>

#include "distributed_matrix.h"
#include "context_handler.h"
#include "matrix_operations.h"
#include "molecular.h"
#include "logging.h"

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
  int rank = MatrixOperations::pseudo_inverse(S, Sinv, 1e-12, BH);
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
  DistributedMatrix::Matrix<double> DxK(naux, nthc, BH.Square);
  MatrixOperations::product(BxL, Sinv, DxK);
  MatrixOperations::redistribute(DxK, BH.Square, BH.Root);
  MatrixOperations::redistribute(BxL, BH.Square, BH.Root);
  if (BH.rank == 0) {
    REQUIRE(!MatrixOperations::is_zero(DxK.store));
    auto it = std::max_element(std::begin(DxK.store), std::end(DxK.store));
    std::cout << *it << std::endl;
    it = std::max_element(std::begin(BxL.store), std::end(BxL.store));
    std::cout << *it << std::endl;
  }
}

TEST_CASE("test_find_pseudo_inverse", "[molecular]")
{
  ContextHandler::BlacsHandler BH;
  // Assuming all data from hdf5 is C ordered.
  bool row_major_read = true;
  bool parallel_read = false;
  std::string input_file = "eri_thc.h5";
  DistributedMatrix::Matrix<double> orbs(input_file, "etaPm", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack which expects fortran order
    MatrixOperations::local_transpose(orbs);
    MatrixOperations::swap_dims(orbs);
  }
  int nthc = orbs.nrows;
  int norb = orbs.ncols;
  MatrixOperations::redistribute(orbs, BH.Root, BH.Square, true);
  DistributedMatrix::Matrix<double> Sinv(nthc, nthc, BH.Square);
  find_pseudo_inverse(orbs, BH, Sinv);
  // X X^+ X = X
  DistributedMatrix::Matrix<double> S(nthc, nthc, BH.Square);
  MatrixOperations::product(orbs, orbs, S, 'N', 'T');
  MatrixOperations::hadamard_product(S);
  DistributedMatrix::Matrix<double> T1(nthc, nthc, BH.Square), T2(nthc, nthc, BH.Square);
  MatrixOperations::product(Sinv, S, T1);
  MatrixOperations::product(S, T1, T2);
  MatrixOperations::redistribute(T2, BH.Square, BH.Root);
  // Reference numbers
  MatrixOperations::redistribute(S, BH.Square, BH.Root);
  DistributedMatrix::Matrix<double> Sinv_ref(input_file, "Sinv", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    REQUIRE(MatrixOperations::normed_difference(T2.store, S.store) == Approx(0.0).margin(1e-10));
    REQUIRE_THAT(T2.store, Catch::Approx<double>(S.store).margin(1e-10));
  }
  MatrixOperations::redistribute(T2, BH.Root, BH.Square);
  // Reference numbers
  MatrixOperations::redistribute(S, BH.Root, BH.Square);
  // X^+ X X^+ = X
  //MatrixOperations::product(S, Sinv, T1);
  //MatrixOperations::product(Sinv, T1, T2);
  //MatrixOperations::redistribute(T2, BH.Square, BH.Root);
  //// Reference numbers
  //MatrixOperations::redistribute(Sinv, BH.Square, BH.Root);
  //if (BH.rank == 0) {
    //REQUIRE_THAT(T2.store, Catch::Approx<double>(Sinv.store).margin(1e-10));
  //}
  //MatrixOperations::redistribute(T2, BH.Root, BH.Square);
  //// Reference numbers
  //MatrixOperations::redistribute(Sinv, BH.Root, BH.Square);
}

TEST_CASE("test_bxl", "[molecular]")
{
  ContextHandler::BlacsHandler BH;
  bool row_major_read = true;
  bool parallel_read = false;
  std::string input_file = "eri_thc.h5";
  DistributedMatrix::Matrix<double> orbs(input_file, "etaPm", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack which expects fortran order
    MatrixOperations::local_transpose(orbs);
    MatrixOperations::swap_dims(orbs);
  }
  MatrixOperations::redistribute(orbs, BH.Root, BH.Square, true);
  DistributedMatrix::Matrix<double> BxL;
  construct_BxL(input_file, orbs, BH, BxL);
  if (BH.rank == 3) {
    REQUIRE(BxL.local_ncols == 64);
    REQUIRE(BxL.local_nrows == 64);
  }
  DistributedMatrix::Matrix<double> BxL_ref(input_file, "T", BH.Root, row_major_read, parallel_read);
  MatrixOperations::redistribute(BxL, BH.Square, BH.Root);
  if (BH.rank == 0) {
    MatrixOperations::local_transpose(BxL, false);
    auto it = std::max_element(std::begin(BxL.store), std::end(BxL.store));
    it = std::max_element(std::begin(BxL_ref.store), std::end(BxL_ref.store));
    //MatrixOperations::local_transpose(BxL);
    double diff = MatrixOperations::normed_difference(BxL.store, BxL_ref.store);
    REQUIRE_THAT(BxL.store, Catch::Approx<double>(BxL_ref.store).margin(1e-10));
    REQUIRE(diff == Approx(0.0).margin(1e-12));
    for (int i = 0; i < BxL.store.size(); i++) {
      if (std::abs(BxL.store[i]-BxL_ref.store[i]) > 1e-8) {
        std::cout << i << " " << BxL.store[i] << " " << BxL_ref.store[i] << std::endl;
      }
    }
  }
}

TEST_CASE("test_orbital_product", "[molecular]")
{
  ContextHandler::BlacsHandler BH;
  bool parallel_read = false;
  bool row_major_read = true;
  std::string input_file = "eri_thc.h5";
  DistributedMatrix::Matrix<double> orbs(input_file, "etaPm", BH.Root, false, parallel_read);
  MatrixOperations::redistribute(orbs, BH.Root, BH.Square, true);
  int msquare = orbs.nrows * orbs.nrows;
  int nbasis = orbs.nrows;
  int nthc = orbs.ncols;
  // Form T[ik,L]
  MatrixOperations::redistribute(orbs, BH.Square, BH.Column, true, nbasis, 1);
  DistributedMatrix::Matrix<double> orb_prod(msquare, nthc, BH.Column, msquare, 1);
  // T[ik,L] = orbs[i,L] orbs[k,L]
  MatrixOperations::tensor_rank_one(orbs, orb_prod);
  MatrixOperations::redistribute(orb_prod, BH.Column, BH.Root);
  DistributedMatrix::Matrix<double> orb_prod_ref(input_file, "orb_prod", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    double diff = MatrixOperations::normed_difference(orb_prod.store, orb_prod_ref.store);
    REQUIRE(diff == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("test_svd", "[molecular]")
{
  ContextHandler::BlacsHandler BH;
  // Assuming all data from hdf5 is C ordered.
  bool row_major_read = true;
  bool parallel_read = false;
  std::string input_file = "eri_thc.h5";
  DistributedMatrix::Matrix<double> orbs(input_file, "etaPm", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack
    MatrixOperations::local_transpose(orbs);
    MatrixOperations::swap_dims(orbs);
  }
  MatrixOperations::redistribute(orbs, BH.Root, BH.Square, true);
  int m = orbs.nrows;
  DistributedMatrix::Matrix<double> S(m, m, BH.Square), U(m,m,BH.Square), VT(m,m,BH.Square);
  MatrixOperations::product(orbs, orbs, S, 'N', 'T');
  MatrixOperations::hadamard_product(S);
  DistributedMatrix::Matrix<double> Sref(S);
  std::vector<double> sigma(S.nrows);
  MatrixOperations::svd(S, U, VT, sigma, BH.Square);
  DistributedMatrix::Matrix<double> SMat(m,m,BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < sigma.size(); i++)
      SMat.store[i+sigma.size()*i] = sigma[i];
  }
  MatrixOperations::redistribute(SMat, BH.Root, BH.Square);
  DistributedMatrix::Matrix<double> T1(m, m, BH.Square), T2(m, m, BH.Square);
  MatrixOperations::product(SMat, VT, T1);
  MatrixOperations::product(U, T1, T2);
  MatrixOperations::redistribute(T2, BH.Square, BH.Root);
  MatrixOperations::redistribute(U, BH.Square, BH.Root);
  MatrixOperations::redistribute(VT, BH.Square, BH.Root);
  //Logging::write_matrix(U, "svd.h5", "U", "", BH.rank==0, true, false, true);
  //Logging::write_matrix(VT, "svd.h5", "VT", "", BH.rank==0, true, false, true);
  // Reference numbers
  MatrixOperations::redistribute(Sref, BH.Square, BH.Root);
  if (BH.rank == 0) {
    REQUIRE_THAT(T2.store, Catch::Approx<double>(Sref.store).margin(1e-14));
  }
}

TEST_CASE("test_multiply", "[molecule]")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> Pinv("numpy_svd.h5", "sinv", BH.Root, true, false);
  if (BH.rank == 0)
  {
    MatrixOperations::local_transpose(Pinv);
    MatrixOperations::swap_dims(Pinv);
  }
  MatrixOperations::redistribute(Pinv, BH.Root, BH.Square, true);
  DistributedMatrix::Matrix<double> BxL("eri_thc.h5", "T", BH.Root, true, false);
  {
    MatrixOperations::local_transpose(BxL);
    MatrixOperations::swap_dims(BxL);
  }
  MatrixOperations::redistribute(BxL, BH.Root, BH.Square);
  DistributedMatrix::Matrix<double> X(BxL.nrows, Pinv.ncols, BH.Square);
  MatrixOperations::product(BxL, Pinv, X);
  MatrixOperations::redistribute(X, BH.Square, BH.Root);
  //DistributedMatrix::Matrix<double> Xref("hamil.h5", "Hamiltonian/THC/Luv", BH.Root, true, false);
  DistributedMatrix::Matrix<double> Xref("eri_thc.h5", "DxK", BH.Root, true, false);
  if (BH.rank == 0) {
    MatrixOperations::local_transpose(X, false);
    for (int i = 0; i < Xref.store.size(); i++) {
      if (std::abs(Xref.store[i]-X.store[i]) > 1e-6) {
        std::cout << i << " " << Xref.store[i] << " " << X.store[i] << std::endl;
      }
    }
    //REQUIRE_THAT(X.store, Catch::Approx<double>(Xref.store).margin(1e-14));
  }
  MatrixOperations::redistribute(BxL, BH.Square, BH.Root);
  MatrixOperations::redistribute(Pinv, BH.Square, BH.Root);
  double one = 1.0, zero = 0.0;
  std::cout << BxL.store[0] << " " << BxL.store[1] << " " << Pinv.store[0] << " " << Pinv.store[1] << std::endl;
  MatrixOperations::product(BxL.nrows, Pinv.ncols, BxL.ncols,
                            one,
                            BxL.store.data(), BxL.nrows,
                            Pinv.store.data(), Pinv.nrows,
                            zero,
                            X.store.data(), X.nrows);
  std::cout << X.store[0] << std::endl;
}

TEST_CASE("test_generate_thc", "[molecular]")
{
  generate_qmcpack_molecular_thc("eri_thc.h5", "hamil_test.h5", true, true);
}
