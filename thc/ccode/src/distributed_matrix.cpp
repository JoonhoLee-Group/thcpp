#include <vector>
#include <mpi.h>
#include <time.h>
#include "distributed_matrix.h"
#include "scalapack_defs.h"
#include "cblacs_defs.h"
#include "H5Cpp.h"
#include "h5helper.h"
#include "utils.h"
#include "context_handler.h"

namespace DistributedMatrix
{
  Matrix::Matrix()
  {
    // Hardcoded for now.
    block_nrows = 64;
    block_ncols = 64;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    desc.resize(9);
  }
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
                 ContextHandler::BlacsGrid &Grid, bool row_major)
  {
    std::vector<hsize_t> dims(2);
    if (Grid.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << " * Reading " << name << " matrix." << std::endl;
      double tread = clock();
      H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);
      // Read data from file.
      H5Helper::read_matrix(file, name, store, dims);
      tread = clock() - tread;
      std::cout << " * Time taken to read matrix: " << " " << tread / CLOCKS_PER_SEC << " seconds" << std::endl;
      double memory = UTILS::get_memory(store);
      std::cout << " * Memory usage for " << name << ": " << memory << " GB" << std::endl;
      file.close();
    }
    MPI_Bcast(dims.data(), 2, MPI::UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (row_major) {
      // Matrices are in native C order.
      nrows = dims[0];
      ncols = dims[1];
      if (Grid.rank == 0) {
        std::cout << " * Assuming matrices are in C / row major format." << std::endl;
      }
    } else {
      // Matrices have been transposed to Fortran ordering before reading from hdf5 so need
      // to swap dimensions for rows and columns.
      if (Grid.rank == 0) {
        std::cout << " * Assuming matrices are in FORTRAN / column major format." << std::endl;
      }
      nrows = dims[1];
      ncols = dims[0];
    }
    if (Grid.rank == 0) {
      std::cout << " * Matrix shape: (" << nrows << ", " << ncols << ")" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
    }
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
    int irsrc = 0, icsrc = 0;
    // 1x1 grid.
    if (Grid.nprocs == 1) {
      Cblacs_gridinfo(Grid.ctxt, &Grid.nrows, &Grid.ncols, &Grid.row, &Grid.col);
      if (Grid.row == 0 && Grid.col == 0) {
        local_nr = nrows;
        local_nc = ncols;
        descinit_(desc.data(), &nrows, &ncols, &nrows,
                  &ncols, &irsrc, &icsrc, &Grid.ctxt, &nrows,
                  &info);
      } else {
        desc[1] = -1;
      }
    } else {
      local_nr = numroc_(&nrows, &block_nrows, &Grid.row,
                         &izero, &Grid.nrows);
      local_nc = numroc_(&ncols, &block_ncols, &Grid.col, &izero,
                         &Grid.ncols);
      //if (Grid.rank == 0) {
        //std::cout << "descinit: " << local_nr << " " << local_nc << " " << nrows << " " << ncols << std::endl;
      //}
      lld = std::max(1, local_nr);
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
    descb.swap(desc);
#ifndef NDEBUG
    if (GridB.rank == 0) {
      double memory = UTILS::get_memory(store);
      std::cout << "#################################################" << std::endl;
      std::cout << " * Local memory usage (on root processor) following redistribution: " << memory << " GB" << std::endl;
      std::cout << "#################################################" << std::endl;
    }
#endif
  }
  void Matrix::setup_matrix(int m, int n, ContextHandler::BlacsGrid &Grid)
  {
    nrows = m;
    ncols = n;
    initialise_discriptor(desc, Grid, local_nrows, local_ncols);
    // Allocate memory.
    store.resize(local_nrows*local_ncols);
  }
  // Destructor.
  // Will need to deal with releasing memory.
  Matrix::~Matrix()
  {
  }
}
