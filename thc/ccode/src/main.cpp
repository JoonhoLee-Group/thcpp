#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <mpi.h>
#include "h5helper.h"
#include "matrix_operations.h"
#include "cblacs_defs.h"

int main(int argc, char* argv[])
{
  int rank, nprocs;
  //std::cout << "HERE" << std::endl;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //std::cout << "MPI INIT" << std::endl;

  bool root = (rank == 0);
  double sum1 = 0, sum2 = 0;
  const double one = 1.0, zero = 0.0;
  bool row_major;
  int proc_rows = 2, proc_cols = 2;
  int block_rows = 100, block_cols = 100;
  int myid, myrow, mycol, numproc, ctxt, ctxt_sys, ctxt_root;
  ctxt = ctxt_sys;
  ctxt_root = ctxt_sys;
  // Initialise blacs context.
  Cblacs_pinfo(&myid, &numproc);
  Cblacs_get(0, 0, &ctxt_sys);
  // Our actual processor distribution.
  Cblacs_gridinit(&ctxt, "Row-major", proc_rows, proc_cols);
  // Initalise grid of size 1 on root so we can reduce distributed matrices.
  Cblacs_gridinit(&ctxt_root, "Row-major", 1, 1);
  DistributedMatrix::Matrix CZt("thc_data.h5", "CZt", block_rows,
                                block_cols, ctxt, ctxt_root, rank);
  DistributedMatrix::Matrix CCt("thc_data.h5", "CCt", block_rows,
                                block_cols, ctxt, ctxt_root, rank);

  //double mem_CCt = UTILS::get_memory(CCt); 
  //double mem_CZt = UTILS::get_memory(CZt); 
  //std::cout << "Memory usage for CCt: " << mem_CCt << " GB" << std::endl;
  //std::cout << "Memory usage for CZt: " << mem_CZt << " GB" << std::endl;
  //std::cout << "Total memory usage: " << mem_CCt + mem_CZt << " GB" << std::endl;
  //double tlsq = clock();
  //std::cout << "Performing serial least squares solve." << std::endl;
  //MatrixOperations::least_squares(CCt.data(), CZt.data(), nmu, nmu, ngrid);
  //tlsq = clock() - tlsq;
  //std::cout << "Time for serial least squares solve : " << tlsq / CLOCKS_PER_SEC << " seconds" << std::endl;
  //H5Helper::write_interpolating_points(CZt, nmu, ngrid);
  MPI_Finalize();
}
