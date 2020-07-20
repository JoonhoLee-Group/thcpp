#include "catch.hpp"
#include "distributed_matrix.h"
#include "context_handler.h"
#include "matrix_operations.h"

TEST_CASE("test_svd_double")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> X(4,4,BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < 16; i++) {
      X.store[i] = i;
    }
  }
  MatrixOperations::redistribute(X, BH.Root, BH.Square, false, 2, 2);
  DistributedMatrix::Matrix<double> U(4,4,BH.Square,2,2), VT(4,4,BH.Square,2,2);
  REQUIRE(X.local_nrows == 2);
  std::vector<double> S(4);
  MatrixOperations::svd(X, U, VT, S, BH.Square);
  MPI_Barrier(MPI_COMM_WORLD);
  // numpy.linalg.svd
  std::vector<double> Sref = {3.513996365902e+01, 2.276610208715e+00, 0.0, 0.0};
  REQUIRE_THAT(S, Catch::Approx<double>(Sref).margin(1e-12));

  // Test factorization X = U S VT
  DistributedMatrix::Matrix<double> SMat(4,4,BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < 4; i++)
      SMat.store[i+4*i] = S[i];
  }
  MatrixOperations::redistribute(SMat, BH.Root, BH.Square, false, 2, 2);
  DistributedMatrix::Matrix<double> T1(4,4,BH.Square,2,2), T2(4,4,BH.Square,2,2);
  MatrixOperations::product(SMat, VT, T1);
  MatrixOperations::product(U, T1, T2);
  MatrixOperations::redistribute(T2, BH.Square, BH.Root);
  MatrixOperations::redistribute(X, BH.Square, BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < 16; i++) {
      X.store[i] = i;
    }
  }
  if (BH.rank == 0) {
    REQUIRE_THAT(X.store, Catch::Approx<double>(T2.store).margin(1e-12));
  }
}

TEST_CASE("test_pseudo_inverse")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> X(4,4,BH.Root,2,2), Xpinv(4,4,BH.Square,2,2);
  if (BH.rank == 0) {
    for (int i = 0; i < 16; i++) {
      X.store[i] = i;
    }
  }
  MatrixOperations::redistribute(X, BH.Root, BH.Square, true, 2, 2);
  DistributedMatrix::Matrix<double> Xcopy(X);
  MatrixOperations::pseudo_inverse(X, Xpinv, 1e-12, BH);
  //std::cout << Xpinv.store[0] << std::endl;
  DistributedMatrix::Matrix<double> T1(4,4,BH.Square,2,2), T2(4,4,BH.Square,2,2);
  // X X^+ X = X
  MatrixOperations::product(Xpinv, Xcopy, T1);
  MatrixOperations::product(Xcopy, T1, T2);
  MatrixOperations::redistribute(T2, BH.Square, BH.Root);
  if (BH.rank == 0) {
    std::vector<double> ref(16);
    for (int i = 0; i < 16; i++) {
      ref[i] = i;
    }
    REQUIRE_THAT(T2.store, Catch::Approx<double>(ref).margin(1e-12));
  }
}

TEST_CASE("test_tensor_rank_one")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> T(16,10,BH.Column,16,1), P(4,10,BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < P.ncols; i++) {
      for (int j = 0; j < P.nrows; j++) {
        P.store[i*P.nrows+j] = j;
      }
    }
  }
  MatrixOperations::redistribute(P, BH.Root, BH.Column, true, 4, 1);
  MatrixOperations::tensor_rank_one(P, T);
  MatrixOperations::redistribute(T, BH.Column, BH.Root, true);
  if (BH.rank == 0) {
    std::vector<double> ref = {0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9};
    int mn = P.nrows * P.nrows;
    for (int i = 0; i < 10; i++) {
      std::vector<double> Ti(mn, 0.0);
      std::copy_n(T.store.data()+i*mn, mn, Ti.data());
      REQUIRE_THAT(Ti, Catch::Approx<double>(ref).margin(1e-12));
    }
  }
  //}
}
