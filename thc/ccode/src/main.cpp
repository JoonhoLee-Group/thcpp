#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <complex>
#include <fftw3.h>
#include <mpi.h>
#include "distributed_matrix.h"
#include "matrix_operations.h"
#include "h5helper.h"
#include "utils.h"
#include "kmeans.h"
#include "interpolating_vectors.h"

int main(int argc, char* argv[])
{
  int rank, nprocs;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Initialise blacs context.
  ContextHandler::BlacsHandler BH;
  std::vector<int> interp_indxs;
  DistributedMatrix::Matrix<double> aoR;
  int max_it = 200;
  double threshold = 1e-3;
  int cfac = 5;
  double sim_time = clock();
  // 1. Determine interpolating points using Veronoi tesselation / KMeans.
  InterpolatingPoints::KMeans KMeansSolver("supercell_atomic_orbitals.h5", max_it, threshold, cfac);
  KMeansSolver.kernel(BH, interp_indxs, aoR);
  //// 2. Determine interpolating vectors via least squares.
  InterpolatingVectors::IVecs IVSolver("supercell_atomic_orbitals.h5", "fcidump.h5", BH, interp_indxs, aoR);
  IVSolver.kernel(BH);
  if (BH.rank == 0) std::cout << " * Total simulation time : " << (clock()-sim_time) / CLOCKS_PER_SEC << " seconds." << std::endl;
  MPI_Finalize();
}
