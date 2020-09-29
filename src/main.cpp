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
#include "interpolating_vectors.h"
#include "interpolating_points.h"

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
  bool half_rotate;
  std::vector<int> interp_indxs;
  UTILS::parse_simple_opts(input_data, BH.rank, thc_cfac, thc_half_cfac, half_rotate);
  // 1. Determine interpolating points for full orbital set.
  InterpolatingPoints::IPoints IPSolver(input_data, BH);
  interp_indxs = IPSolver.kernel(input_data, BH, thc_cfac, false);
  // 2. Determine interpolating vectors via least squares.
  {
    // Half rotate
    bool half_rotate_first = false;
    // Create hdf5 file first time around.
    bool open_file = true;
    InterpolatingVectors::IVecs IVSolver(input_data, BH, interp_indxs, half_rotate_first, open_file);
    IVSolver.kernel(BH);
    if (BH.rank == 0 && !half_rotate) IVSolver.dump_qmcpack_data(thc_cfac, thc_half_cfac, BH);
  }
  if (half_rotate) {
    if (thc_half_cfac != thc_cfac) {
      InterpolatingPoints::IPoints IP(input_data, BH);
      interp_indxs = IP.kernel(input_data, BH, thc_half_cfac, half_rotate);
    }
    bool open_file = false;
    InterpolatingVectors::IVecs IVSolver(input_data, BH, interp_indxs, half_rotate, open_file);
    IVSolver.kernel(BH);
    if (BH.rank == 0) IVSolver.dump_qmcpack_data(thc_cfac, thc_half_cfac, BH);
  }
  if (BH.rank == 0) {
    std::cout << " * Total simulation time : " << (clock()-sim_time) / CLOCKS_PER_SEC << " seconds." << std::endl;
  }
  MPI_Finalize();
}
