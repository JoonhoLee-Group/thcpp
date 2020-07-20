#include "catch.hpp"
#include <iostream>
#include <mpi.h>

TEST_CASE("TEST", "[scalapack]")
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << rank << std::endl;
  REQUIRE(1==1);
}
