#include <vector>
#include <mpi.h>
#include <time.h>
#include "distributed_matrix.h"
#include "scalapack_defs.h"
#include "H5Cpp.h"
#include "h5helper.h"
#include "utils.h"
#include "context_handler.h"

namespace DistributedMatrix
{
  // Constructor without having data.
  Matrix::Matrix(int m, int n, ContextHandler::BlacsGrid &Grid)
  {
    nrows = m;
    ncols = n;
    // Hardcoded for now.
    block_nrows = 64;
    block_ncols = 64;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    desc.resize(9);
    initialise_discriptor(desc, Grid, local_nrows, local_ncols);
    // Allocate memory.
    store.resize(local_nrows*local_ncols);
  }

  Matrix::Matrix(std::string filename, std::string name,
                 ContextHandler::BlacsGrid &Grid, int rank)
  {
    std::vector<hsize_t> dims(2);
    if (rank == 0) {
      std::cout << "Reading " << name << " matrix." << std::endl;
      double tread = clock();
      H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);
      // Read data from file.
      H5Helper::read_matrix(file, name, store, dims);
      tread = clock() - tread;
      std::cout << "Time taken to read matrix: " << " " << tread / CLOCKS_PER_SEC << " seconds" << std::endl;
      double memory = UTILS::get_memory(store);
      std::cout << "Memory usage for " << name << ": " << memory << " GB" << std::endl;
      std::cout << "Assuming matrices are in FORTRAN / column major format." << std::endl;
      std::cout << "Matrix shape: (" << dims[1] << ", " << dims[0] << ")" << std::endl;
      file.close();
    }
    MPI_Bcast(dims.data(), 2, MPI::UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    // Matrices have been transposed to Fortran ordering before reading from hdf5 so need
    // to swap dimensions for rows and columns.
    nrows = dims[1];
    ncols = dims[0];
    // Hardcoded.
    block_nrows = 64;
    block_ncols = 64;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    desc.resize(9);
    initialise_discriptor(desc, Grid, local_nrows, local_ncols);
  }
  void Matrix::initialise_discriptor(std::vector<int> &desc, ContextHandler::BlacsGrid &Grid, int &local_nr, int &local_nc)
  {
    local_nr = numroc_(&nrows, &block_nrows, &Grid.row,
                       &izero, &Grid.nrows);
    local_nc = numroc_(&ncols, &block_ncols, &Grid.col, &izero,
                       &Grid.ncols);
    lld = std::max(1, local_nr);
    int irsrc = 0, icsrc = 0;
    // 1x1 grid.
    if (Grid.nrows == 1 && Grid.ncols == 1) {
      if (Grid.row == 0 && Grid.col == 1) {
        descinit_(desc.data(), &nrows, &ncols, &block_nrows,
                  &block_ncols, &irsrc, &icsrc, &Grid.ctxt, &lld,
                  &info);
      } else {
        desc[1] = -1;
      }
    } else {
      descinit_(desc.data(), &nrows, &ncols, &block_nrows,
                &block_ncols, &irsrc, &icsrc, &Grid.ctxt, &lld,
                &info);
    }
  }
  void Matrix::redistribute(ContextHandler::BlacsGrid &GridA, ContextHandler::BlacsGrid &GridB)
  {
    // setup discriptor for Blacs grid we'll distribute to.
    std::vector<int> descb(9);
    int nr, nc;
    initialise_discriptor(descb, GridB, nr, nc);
    std::vector<double> tmp(nr*nc);
    int ctxt;
    // ctxt for p?gemr2d call must at least contain the union of processors from gridA and
    // gridB.
    if (GridA.nrows*GridA.ncols >= GridB.nrows*GridB.ncols) {
      ctxt = GridA.ctxt;
    } else {
      ctxt = GridB.ctxt;
    }
    pdgemr2d_(&nrows, &ncols,
              store.data(), &init_row_idx, &init_row_idx, desc.data(),
              tmp.data(), &init_row_idx, &init_col_idx, descb.data(),
              &ctxt);
    tmp.swap(store);
#ifdef DEBUG
    if (GridB.rank == 0) {
      double memory = UTILS::get_memory(store);
      std::cout << "Local memory usage (on root processor) for " << name << ": " << memory << " GB" << std::endl;
    }
#endif
  }
  // Destructor.
  // Will need to deal with releasing memory.
  Matrix::~Matrix()
  {
  }
}
