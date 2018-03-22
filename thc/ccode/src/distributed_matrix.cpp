#include <vector>
#include "distributed_matrix.h"
#include "cblacs_defs.h"
#include "scalapack_defs.h"

namespace DistributedMatrix
{
  // Constructor.
  Matrix::Matrix(int m, int n, int block_m, int block_n, int ctxt, int root_ctxt)
  {
    nrows = m;
    ncols = n;
    block_nrows = block_m;
    block_ncols = block_n;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
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
    // We fake a 1x1 processor grid.
    desc_global[1] = -1;
    if (proc_row == 0 && proc_col == 0) {
      descinit_(desc_global.data(), &nrows, &ncols, &nrows,
                &ncols, &irsrc, &icsrc, &root_ctxt, &nrows, &info);
    }
    // Allocate memory.
    global_data.resize(nrows*ncols);
    local_data.resize(local_nrows*local_ncols);
  }
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
