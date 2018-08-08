#include "context_handler.h"
#include <math.h>
#include <iostream>
#include <mpi.h>

namespace ContextHandler
{
  BlacsHandler::BlacsHandler()
  {
    Cblacs_pinfo(&rank, &nprocs);
    proc_rows = (int)sqrt(nprocs);
    proc_cols = nprocs / proc_rows;
    // Initialise differenct blacs grids.
    Root = BlacsGrid(1, 1, rank);
    Square = BlacsGrid(proc_rows, proc_cols, rank);
    Column = BlacsGrid(1, proc_rows*proc_cols, rank);
    comm = MPI_COMM_WORLD;
    if (proc_rows != proc_cols && rank == 0) {
      std::cout << "Not running on square processor grid." << std::endl;
    }
  }
  BlacsHandler::~BlacsHandler() {}
  // Constructor for potentially non-square processor grid.
  BlacsGrid::BlacsGrid(int nr, int nc, int rk)
  {
    nrows = nr;
    ncols = nc;
    nprocs = nr*nc;
    rank = rk;
    int ctxt_sys;
    Cblacs_get(0, 0, &ctxt_sys);
    ctxt = ctxt_sys;
    Cblacs_gridinit(&ctxt, "Row-major", nrows, ncols);
    Cblacs_gridinfo(ctxt, &nrows, &ncols, &row, &col);
    comm = MPI_COMM_WORLD;
  }
  BlacsGrid::~BlacsGrid() {}
}
