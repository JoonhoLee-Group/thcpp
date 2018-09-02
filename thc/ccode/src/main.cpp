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
#include "qrcp.h"

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
  double sim_time = clock();
  nlohmann::json input_data;
  if (rank == 0) {
    std::ifstream input_file(argv[1]);
    input_file >> input_data;
    UTILS::print_header(nprocs, input_data);
  }
  int thc_cfac, thc_half_cfac;
  bool half_rotated = false;
  UTILS::parse_simple_opts(input_data, rank, thc_cfac, thc_half_cfac, half_rotated);
  // 1. Determine interpolating points using Veronoi tesselation / KMeans.
  std::vector<int> interp_indxs;
  //{
    //InterpolatingPoints::KMeans KMeansSolver(input_data, thc_cfac, BH);
    //KMeansSolver.kernel(BH, interp_indxs);
  //}
  {
    QRCP::QRCPSolver QRCPSolver(input_data, thc_cfac, BH);
    QRCPSolver.kernel(BH, interp_indxs);
  }
  // 2. Determine interpolating vectors via least squares.
  {
    InterpolatingVectors::IVecs IVSolver(input_data, BH, interp_indxs, false, true);
    IVSolver.kernel(BH);
  }
  //if (half_rotated) {
    //if (thc_half_cfac != thc_cfac) {
      //InterpolatingPoints::KMeans KMeansSolver(input_data, thc_half_cfac, BH);
      //KMeansSolver.kernel(BH, interp_indxs);
    //}
    //InterpolatingVectors::IVecs IVSolver(input_data, BH, interp_indxs, true, false);
    //IVSolver.kernel(BH);
    //if (BH.rank == 0) IVSolver.dump_qmcpack_data(thc_cfac, thc_half_cfac, BH);
  //}
  if (BH.rank == 0) {
    // Simple qmcpack data (hcore, dimensions etc.)
    std::cout << " * Total simulation time : " << (clock()-sim_time) / CLOCKS_PER_SEC << " seconds." << std::endl;
  }
  MPI_Finalize();
}
