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
  Matrix::Matrix(int m, int n, int block_m, int block_n, int &ctxt, int &root_ctxt, int &ccyc_ctxt)
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
    initialise_discriptors(ctxt, root_ctxt, ccyc_ctxt);
    // Allocate memory.
    local_data.resize(local_nrows*local_ncols);
    ccyc_data.resize(nrows*ccyc_ncols);
    fft_data.resize(nrows*ccyc_ncols);
  }
  Matrix::Matrix(std::string filename, std::string name, int block_m, int block_n,
                 int &ctxt, int &root_ctxt, int &ccyc_ctxt, int rank)
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
      std::cout << "Assuming matrices are in FORTRAN / column major format." << std::endl;
      std::cout << "Matrix shape: (" << dims[1] << ", " << dims[0] << ")" << std::endl;
      file.close();
    }
    MPI_Bcast(dims.data(), 2, MPI::UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    // Matrices have been transposed to Fortran ordering before reading from hdf5 so need
    // to swap dimensions for rows and columns.
    nrows = dims[1];
    ncols = dims[0];
    block_nrows = block_m;
    block_ncols = block_n;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    initialise_discriptors(ctxt, root_ctxt, ccyc_ctxt);
    // Store for local arrays.
    local_data.resize(local_nrows*local_ncols);
    ccyc_data.resize(nrows*ccyc_ncols);
    if (rank == 0) {
      double memory = UTILS::get_memory(local_data);
      std::cout << "Local memory usage (on root processor) for " << name << ": " << memory << " GB" << std::endl;
    }
  }
  // Initialise descriptor arrays for block cyclic distributions.
  void Matrix::initialise_discriptors(int ctxt, int root_ctxt, int ccyc_ctxt)
  {
    // Query location in processor grid.
    Cblacs_gridinfo(ctxt, &proc_nrows, &proc_ncols, &proc_row, &proc_col);
    local_nrows = numroc_(&nrows, &block_nrows, &proc_row, &izero, &proc_nrows);
    local_ncols = numroc_(&ncols, &block_ncols, &proc_col, &izero, &proc_ncols);
    // Local leading dimension.
    lld = std::max(1, local_nrows);
    desc_global.resize(9);
    desc_local.resize(9);
    desc_ccyc.resize(9);
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
    // column cyclic distribution.
    int ncols_per_block = 1;
    int ccyc_proc_nrows, ccyc_proc_ncols, ccyc_proc_row, ccyc_proc_col;
    Cblacs_gridinfo(ccyc_ctxt, &ccyc_proc_nrows, &ccyc_proc_ncols, &ccyc_proc_row, &ccyc_proc_col);
    ccyc_ncols = numroc_(&ncols, &ncols_per_block, &ccyc_proc_row, &izero, &ccyc_proc_ncols);
    ccyc_nrows = numroc_(&nrows, &nrows, &ccyc_proc_row, &izero, &ccyc_proc_nrows);
    descinit_(desc_ccyc.data(), &nrows, &ncols, &nrows, &ccyc_ncols,
              &irsrc, &icsrc, &ccyc_ctxt, &nrows, &info);

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
  void Matrix::redistribute_to_column_cyclic(int ctxt)
  {
    pdgemr2d_(&nrows, &ncols,
              local_data.data(), &init_row_idx, &init_row_idx, desc_local.data(),
              ccyc_data.data(), &init_row_idx, &init_col_idx, desc_ccyc.data(),
              &ctxt);
  }
  void Matrix::gather_fft(int ctxt)
  {
    if (cglobal_data.size() == 0) cglobal_data.resize(nrows*ncols);
    pzgemr2d_(&nrows, &ncols,
              fft_data.data(), &init_row_idx, &init_row_idx, desc_ccyc.data(),
              cglobal_data.data(), &init_row_idx, &init_col_idx, desc_global.data(),
              &ctxt);
  }
  // Destructor.
  // Will need to deal with releasing memory.
  Matrix::~Matrix()
  {
  }
}
