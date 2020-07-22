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

TEST_CASE("test_hadamard")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> T(4,4,BH.Square,2,2), T2(4,4,BH.Square,2,2);
  for (int i = 0; i < T.store.size(); i++) {
    T.store[i] = 2.0;
    T2.store[i] = 4.0;
  }
  MatrixOperations::hadamard_product(T);
  REQUIRE_THAT(T2.store, Catch::Approx<double>(T.store).margin(1e-12));
}

TEST_CASE("test_transpose")
{
  ContextHandler::BlacsHandler BH;
  //SECTION("fortran_order")
  {
    DistributedMatrix::Matrix<double> T1(4,2,BH.Root);
    DistributedMatrix::Matrix<double> T2(2,4,BH.Root);
    if (BH.rank == 0) {
      std::vector<double> x = {1,2,3,4,5,6,7,8};
      std::vector<double> y = {1,5,2,6,3,7,4,8};
      T1.store = x;
      T2.store = y;
      MatrixOperations::local_transpose(T1, false);
      REQUIRE_THAT(T1.store, Catch::Approx<double>(y).margin(1e-12));
      MatrixOperations::local_transpose(T1, false);
      REQUIRE_THAT(T1.store, Catch::Approx<double>(x).margin(1e-12));
    }
  }
  SECTION("c_order")
  {
    DistributedMatrix::Matrix<double> T1(4,2,BH.Root);
    DistributedMatrix::Matrix<double> T2(2,4,BH.Root);
    if (BH.rank == 0) {
      std::vector<double> x = {1,2,3,4,5,6,7,8};
      std::vector<double> y = {1,3,5,7,2,4,6,8};
      T1.store = x;
      T2.store = y;
      MatrixOperations::local_transpose(T1);
      REQUIRE_THAT(T1.store, Catch::Approx<double>(y).margin(1e-12));
      MatrixOperations::local_transpose(T1);
      REQUIRE_THAT(T1.store, Catch::Approx<double>(x).margin(1e-12));
    }
  }
  SECTION("parallel")
  {
    DistributedMatrix::Matrix<double> T1(4,2,BH.Root);
    DistributedMatrix::Matrix<double> T2(2,4,BH.Square);
    std::vector<double> x = {1,2,3,4,5,6,7,8};
    std::vector<double> y = {1,5,2,6,3,7,4,8};
    if (BH.rank == 0) {
      T1.store = x;
    }
    MatrixOperations::redistribute(T1, BH.Root, BH.Square);
    MatrixOperations::transpose(T1, T2);
    MatrixOperations::redistribute(T2, BH.Square, BH.Root);
    if (BH.rank == 0) {
      REQUIRE_THAT(T2.store, Catch::Approx<double>(y).margin(1e-12));
    }
    MatrixOperations::redistribute(T2, BH.Root, BH.Square);
    MatrixOperations::transpose(T2, T1);
    MatrixOperations::redistribute(T1, BH.Square, BH.Root);
    if (BH.rank == 0) {
      REQUIRE_THAT(T1.store, Catch::Approx<double>(x).margin(1e-12));
    }
  }
}

TEST_CASE("test_product")
{
  ContextHandler::BlacsHandler BH;
  {
    DistributedMatrix::Matrix<double> T1(4,2,BH.Root);
    DistributedMatrix::Matrix<double> T2(2,4,BH.Root);
    DistributedMatrix::Matrix<double> T3(4,2,BH.Root);
    DistributedMatrix::Matrix<double> T4(4,4,BH.Square);
    std::vector<double> x = {1,2,3,4,5,6,7,8};
    std::vector<double> y = {1,7,9,4,5,6,7,8};
    std::vector<double> z = {1,9,5,7,7,4,6,8};
    // note fortran order
    std::vector<double> ref = {36, 44, 52, 60,
                               29, 42, 55, 68,
                               35, 46, 57, 68,
                               47, 62, 77, 92};
    if (BH.rank == 0) {
      T1.store = x;
      T2.store = y;
      T3.store = z;
    }
    MatrixOperations::redistribute(T1, BH.Root, BH.Square);
    MatrixOperations::redistribute(T2, BH.Root, BH.Square);
    MatrixOperations::redistribute(T3, BH.Root, BH.Square);
    SECTION("AxB_NN")
    {
      MatrixOperations::product(T1, T2, T4, 'N', 'N');
      MatrixOperations::redistribute(T4, BH.Square, BH.Root);
      if (BH.rank == 0) {
        REQUIRE_THAT(T4.store, Catch::Approx<double>(ref).margin(1e-12));
      }
      MatrixOperations::redistribute(T4, BH.Root, BH.Square);
    }
    SECTION("AxB_NT")
    {
      MatrixOperations::product(T1, T3, T4, 'N', 'T');
      MatrixOperations::redistribute(T4, BH.Square, BH.Root);
      if (BH.rank == 0) {
        REQUIRE_THAT(T4.store, Catch::Approx<double>(ref).margin(1e-12));
      }
      MatrixOperations::redistribute(T4, BH.Root, BH.Square);
    }
    SECTION("BxA_TT")
    {
      std::vector<double> ref_T = {36, 29, 35, 47,
                                   44, 42, 46, 62,
                                   52, 55, 57, 77,
                                   60, 68, 68, 92};
      MatrixOperations::product(T2, T1, T4, 'T', 'T');
      MatrixOperations::redistribute(T4, BH.Square, BH.Root);
      if (BH.rank == 0) {
        REQUIRE_THAT(T4.store, Catch::Approx<double>(ref_T).margin(1e-12));
      }
      MatrixOperations::redistribute(T4, BH.Root, BH.Square);
    }
    SECTION("BxA_TN")
    {
      DistributedMatrix::Matrix<double> T5(2,2,BH.Square);
      MatrixOperations::product(T3, T1, T5, 'T', 'N');
      MatrixOperations::redistribute(T5, BH.Square, BH.Root);
      std::vector<double> ref_tn = {62, 65, 150, 165};
      if (BH.rank == 0) {
        REQUIRE_THAT(T5.store, Catch::Approx<double>(ref_tn).margin(1e-12));
      }
      MatrixOperations::redistribute(T5, BH.Root, BH.Square);
    }
  }
}
