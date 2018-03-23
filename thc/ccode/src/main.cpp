#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include <mpi.h>
#include "h5helper.h"
#include "matrix_operations.h"
#include "cblacs_defs.h"
#include "fftw3.h"

int main(int argc, char* argv[])
{
  int rank, nprocs;
  //std::cout << "HERE" << std::endl;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //std::cout << "CZT: " << std::endl;
  //std::cout << "MPI INIT" << std::endl;

  bool root = (rank == 0);
  double sum1 = 0, sum2 = 0;
  const double one = 1.0, zero = 0.0;
  bool row_major;
  int proc_rows = 2, proc_cols = 2;
  int block_rows = 64, block_cols = 64;
  int myid, myrow, mycol, numproc, ctxt, ctxt_sys, root_ctxt;
  // Initialise blacs context.
  Cblacs_pinfo(&myid, &numproc);
  //std::cout << "PINFO: " << std::endl;
  Cblacs_get(0, 0, &ctxt_sys);
  ctxt = ctxt_sys;
  root_ctxt = ctxt_sys;
  //std::cout << "GET: " << std::endl;
  // Our actual processor distribution.
  Cblacs_gridinit(&ctxt, "Row-major", proc_rows, proc_cols);
  //std::cout << "INIT: " << std::endl;
  // Initalise grid of size 1 on root so we can reduce distributed matrices.
  Cblacs_gridinit(&root_ctxt, "Row-major", 1, 1);
  //std::cout << "ROOT: " << std::endl;
  //std::cout << "CZT: " << std::endl;
  DistributedMatrix::Matrix CZt("thc_data.h5", "CZt", block_rows,
                                block_cols, ctxt, root_ctxt, rank);
  DistributedMatrix::Matrix CCt("thc_data.h5", "CCt", block_rows,
                                block_cols, ctxt, root_ctxt, rank);
  CZt.scatter_block_cyclic(ctxt);
  CCt.scatter_block_cyclic(ctxt);

  double tlsq = clock();
  if (root) {
    std::cout << "Performing least squares solve." << std::endl;
    std::cout << MatrixOperations::vector_sum(CZt.global_data) << " " << MatrixOperations::vector_sum(CCt.global_data) << " " << std::endl;
  }
  //double local_sum, global_sum;
  //CZt.gather_block_cyclic(ctxt);
  //local_sum = MatrixOperations::vector_sum(CZt.local_data);
  //MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  //if (root) std::cout << "REDUCE : " << global_sum << " " << CZt.ncols << std::endl;
  //DistributedMatrix::Matrix C(CCt.nrows, CZt.ncols, block_rows, block_cols, ctxt, root_ctxt); 
  //std::cout << C.nrows << " " << C.ncols << " " << CZt.ncols << " " << CZt.nrows << std::endl;
  //MatrixOperations::product(CCt, CZt, C);
  //C.gather_block_cyclic(ctxt);
  //if (root) std::cout << "MM: " << MatrixOperations::vector_sum(C.global_data) << std::endl;
  MatrixOperations::least_squares(CCt, CZt);
  CZt.gather_block_cyclic(ctxt);
  tlsq = clock() - tlsq;
  if (root) {
    std::cout << "Time for least squares solve : " << tlsq / CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "SUM: " << MatrixOperations::vector_sum(CZt.global_data) << std::endl;
    H5Helper::write_interpolating_points(CZt.global_data, CZt.nrows, CZt.ncols);
  }
  MPI_Finalize();
}
