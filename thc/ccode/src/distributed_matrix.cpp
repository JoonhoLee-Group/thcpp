#include <vector>
#include <mpi.h>
#include <time.h>
#include "distributed_matrix.h"
#include "cblacs_defs.h"
#include "scalapack_defs.h"
#include "H5Cpp.h"
#include "h5helper.h"
#include "utils.h"

namespace DistributedMatrix
{
  // Constructor without having data.
  Matrix::Matrix(int m, int n, int block_m, int block_n, int &ctxt, int &root_ctxt)
  {
    nrows = m;
    ncols = n;
    block_nrows = block_m;
    block_ncols = block_n;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    initialise_discriptors(ctxt, root_ctxt);
    // Allocate memory.
    global_data.resize(nrows*ncols);
    local_data.resize(local_nrows*local_ncols);
  }
  Matrix::Matrix(std::string filename, std::string name, int block_m, int block_n,
                 int &ctxt, int &root_ctxt, int rank)
  {
    std::vector<hsize_t> dims(2);
    if (rank == 0) {
      std::cout << "Reading " << name << " matrix." << std::endl;
      double tread = clock();
      H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);
      // Read data from file.
      H5Helper::read_matrix(file, name, global_data, dims);
      tread = clock() - tread;
      std::cout << "Time taken to read matrix: " << " " << tread / CLOCKS_PER_SEC << " seconds" << std::endl;
      double memory = UTILS::get_memory(global_data);
      std::cout << "Memory usage for " << name << ": " << memory << " GB" << std::endl;
      file.close();
    }
    MPI_Bcast(dims.data(), 2, MPI::UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    nrows = dims[0];
    ncols = dims[1];
    block_nrows = block_m;
    block_ncols = block_n;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    initialise_discriptors(ctxt, root_ctxt);
    // Store for local arrays.
    local_data.resize(local_nrows*local_ncols);
    if (rank == 0) {
      double memory = UTILS::get_memory(local_data);
      std::cout << "Local memory usage (on root processor) for " << name << ": " << memory << " GB" << std::endl;
    }
  }
  // Initialise descriptor arrays for block cyclic distributions.
  void Matrix::initialise_discriptors(int ctxt, int root_ctxt)
  {
    // Query location in processor grid.
    Cblacs_gridinfo(ctxt, &proc_nrows, &proc_ncols, &proc_row, &proc_col);
    local_nrows = numroc_(&nrows, &block_nrows, &proc_row, &izero, &proc_nrows);
    local_ncols = numroc_(&ncols, &block_ncols, &proc_col, &izero, &proc_ncols);
    // Local leading dimension.
    lld = std::max(1, local_nrows);
    desc_global.resize(9);
    desc_local.resize(9);
    // Initialise local discriptor array.
    int irsrc = 0, icsrc = 0;
    descinit_(desc_local.data(), &nrows, &ncols, &block_nrows,
              &block_ncols, &irsrc, &icsrc, &ctxt, &lld, &info);
    // Initialise global descriptor array for array stored on root node.
    if (proc_row == 0 && proc_col == 0) {
      descinit_(desc_global.data(), &nrows, &ncols, &nrows,
                &ncols, &irsrc, &icsrc, &root_ctxt, &nrows, &info);
    } else {
      // This needs to be set to -1 on processors other than the root processor
      // for redistribution to work.
      desc_global[1] = -1;
    }
  };
  // Scatter global matrix from root processor to childred in block cyclic distribution.
  void Matrix::scatter_block_cyclic(int ctxt)
  {
    pdgemr2d_(&nrows, &ncols,
              global_data.data(), &init_row_idx, &init_col_idx, desc_global.data(),
              local_data.data(), &init_row_idx, &init_col_idx, desc_local.data(),
              &ctxt);
  }
  // Gather block-cyclically distributed matrix to master processor/node.
  void Matrix::gather_block_cyclic(int ctxt)
  {
    pdgemr2d_(&nrows, &ncols,
              local_data.data(), &init_row_idx, &init_row_idx, desc_local.data(),
              global_data.data(), &init_row_idx, &init_col_idx, desc_global.data(),
              &ctxt);
  }
  // Destructor.
  // Will need to deal with releasing memory.
  Matrix::~Matrix()
  {
  }
}
