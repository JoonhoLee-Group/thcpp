#include <math.h>
#include <iostream>
#include <mpi.h>

#include "context_handler.h"

namespace ContextHandler
{
  BlacsHandler::BlacsHandler()
  {
    Cblacs_pinfo(&rank, &nprocs);
    proc_rows = (int)sqrt(nprocs);
    proc_cols = nprocs / proc_rows;
    // Initialise differenct blacs grids.
    Root = BlacsGrid(1, 1, rank, root);
    Square = BlacsGrid(proc_rows, proc_cols, rank, square);
    Column = BlacsGrid(1, proc_rows*proc_cols, rank, column);
    comm = MPI_COMM_WORLD;
    int err = 0;
    if (proc_rows != proc_cols && rank == 0) {
      std::cerr << "ERROR : Not running on square processor grid." << std::endl;
      err = 1;
    }
    MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (err != 0) {
        exit(1);
    }
  }
  BlacsHandler::~BlacsHandler() {}
  // Constructor for potentially non-square processor grid.
  BlacsGrid::BlacsGrid(int nr, int nc, int rk, Layouts grid_type)
  {
    nrows = nr;
    ncols = nc;
    nprocs = nr*nc;
    rank = rk;
    layout = grid_type;
    int ctxt_sys;
    Cblacs_get(0, 0, &ctxt_sys);
    ctxt = ctxt_sys;
    Cblacs_gridinit(&ctxt, "Row-major", nrows, ncols);
    Cblacs_gridinfo(ctxt, &nrows, &ncols, &row, &col);
    comm = MPI_COMM_WORLD;
  }
  BlacsGrid::~BlacsGrid() {}
}
