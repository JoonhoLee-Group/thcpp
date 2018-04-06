#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <stdlib.h>
#include <complex>
#include <fftw3.h>
#include <mpi.h>
#include "json.hpp"
#include "distributed_matrix.h"
#include "matrix_operations.h"
#include "h5helper.h"
#include "utils.h"
#include "kmeans.h"
#include "interpolating_vectors.h"

int main(int argc, char** argv)
{
  int rank, nprocs;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  // Initialise blacs context.
  ContextHandler::BlacsHandler BH;
  // Parse input.
  if (argc != 2) {
    std::cout << "Usage: tchpp.x input.json" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream input_file(argv[1]);
  nlohmann::json input_data;
  input_file >> input_data;
  double sim_time = clock();
  // 1. Determine interpolating points using Veronoi tesselation / KMeans.
  InterpolatingPoints::KMeans KMeansSolver(input_data);
  std::vector<int> interp_indxs;
  KMeansSolver.kernel(BH, interp_indxs);
  // 2. Determine interpolating vectors via least squares.
  InterpolatingVectors::IVecs IVSolver(input_data, BH, interp_indxs);
  IVSolver.kernel(BH);
  if (BH.rank == 0) std::cout << " * Total simulation time : " << (clock()-sim_time) / CLOCKS_PER_SEC << " seconds." << std::endl;
  MPI_Finalize();
}
