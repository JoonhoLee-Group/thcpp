#include "catch.hpp"
#include <iostream>
#include <mpi.h>

#include "distributed_matrix.h"
#include "context_handler.h"

TEST_CASE("test_default", "[matrix]")
{
  DistributedMatrix::Matrix<double> X;
}

TEST_CASE("test_dims", "[matrix]")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> X(4,4,BH.Square);
}

TEST_CASE("test_dims_blacs", "[matrix]")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> X(128,128,BH.Square,32,32);
  REQUIRE(X.nrows == 128);
  REQUIRE(X.ncols == 128);
  REQUIRE(X.local_nrows == 64);
  REQUIRE(X.local_nrows == 64);
}

TEST_CASE("test_read_write", "[matrix]")
{
  ContextHandler::BlacsHandler BH;
  DistributedMatrix::Matrix<double> X(4,4,BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < 16; i++) {
      X.store[i] = i;
    }
  }
  if (BH.rank == 0) {
    H5::H5File file = H5::H5File("test.h5", H5F_ACC_TRUNC);
    X.dump_data(file, "matrix", "data");
    file.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  DistributedMatrix::Matrix<double> Y("test.h5", "matrix/data", BH.Root, true, false);
  if (BH.rank == 0) {
    REQUIRE(Y.store[5] == X.store[5]);
    REQUIRE(Y.local_ncols == 4);
  } else {
    REQUIRE(Y.local_ncols == 0);
    REQUIRE(Y.local_nrows == 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  DistributedMatrix::Matrix<double> Z("test.h5", "matrix/data", BH.Column, true, true);
  REQUIRE(Z.local_ncols == 1);
  REQUIRE(Z.local_nrows == 4);
  if (BH.rank == 2) {
    std::vector<double> test = {8.0,9.0,10.0,11.0};
    REQUIRE_THAT(Z.store, Catch::Approx<double>(test));
  }
}
