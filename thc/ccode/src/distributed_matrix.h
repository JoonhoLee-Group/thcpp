#ifndef DISTRIBUTED_MATRIX_H
#define DISTRIBUTED_MATRIX_H
#include <iostream>
#include <vector>
#include <complex>
#include "context_handler.h"
#include "h5helper.h"
#include "scalapack_defs.h"
#include "context_handler.h"
#include "utils.h"

// Operations for matrices distributed using scalapack.
// Asumes column major (Fortran) order for layout.
// Thin wrapper to distribute data to and from block cyclic form.
namespace DistributedMatrix
{
  template <class T>
  class Matrix
  {
    public:
      Matrix();
      Matrix(int nrows, int ncols, ContextHandler::BlacsGrid &Grid);
      Matrix(int nrows, int ncols, ContextHandler::BlacsGrid &Grid, int br, int bc);
      Matrix(std::string filename, std::string name,
             ContextHandler::BlacsGrid &Grid, bool row_major=true);
      Matrix(const Matrix& M);
      ~Matrix();
      void gather_block_cyclic(int ctxt);
      void gather_fft(int ctxt);
      void scatter_block_cyclic(int ctxt);
      void initialise_descriptor(std::vector<int> &desc, ContextHandler::BlacsGrid &Grid, int &nr, int &nc);
      void initialise_descriptor(std::vector<int> &desc, ContextHandler::BlacsGrid &Grid, int &nr, int &nc, int br, int bc);
      void setup_matrix(int m, int c, ContextHandler::BlacsGrid &Grid);
      void dump_data(H5::H5File &fh5, std::string group_name, std::string dataset_name);
      // global matrix dimensions
      int nrows;
      int ncols;
      // Block sizes.
      int block_nrows;
      int block_ncols;
      // local number of rows and columns.
      int local_nrows;
      int local_ncols;
      int ccyc_nrows;
      int ccyc_ncols;
      // local leading dimension.
      int lld;
      // Variables for fancy selection of data. Not used set to 0 or 1.
      int izero;
      int init_row_idx, init_col_idx;
      // Variable for fortran interface.
      int info;
      // Global and local data stores for matrix.
      std::vector<T> store;
      // scalapack descriptor array for current blacs context.
      std::vector<int> desc;
  };
  // Default constructor.
  template <class T>
  Matrix<T>::Matrix()
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
  template <class T>
  Matrix<T>::Matrix(int m, int n, ContextHandler::BlacsGrid &Grid)
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
    initialise_descriptor(desc, Grid, local_nrows, local_ncols);
    // Allocate memory.
    store.resize(local_nrows*local_ncols);
  }

  // Constructor without having data, specifying block size.
  template <class T>
  Matrix<T>::Matrix(int m, int n, ContextHandler::BlacsGrid &Grid, int br, int bc)
  {
    nrows = m;
    ncols = n;
    // Hardcoded for now.
    block_nrows = br;
    block_ncols = bc;
    // Offsets.
    izero = 0;
    init_row_idx = 1; // fortran indexing.
    init_col_idx = 1;
    // Setup descriptor arrays for block cyclic distribution.
    desc.resize(9);
    initialise_descriptor(desc, Grid, local_nrows, local_ncols);
    // Allocate memory.
    store.resize(local_nrows*local_ncols);
  }

  // Read from file.
  template <class T>
  Matrix<T>::Matrix(std::string filename, std::string name,
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
    initialise_descriptor(desc, Grid, local_nrows, local_ncols);
  }

  // setup desc for given blacs context arrays.
  template <class T>
  void Matrix<T>::initialise_descriptor(std::vector<int> &desc, ContextHandler::BlacsGrid &Grid, int &local_nr, int &local_nc)
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
  template <class T>
  void Matrix<T>::initialise_descriptor(std::vector<int> &desc, ContextHandler::BlacsGrid &Grid, int &local_nr, int &local_nc, int br, int bc)
  {
    int irsrc = 0, icsrc = 0;
    block_nrows = br;
    block_ncols = bc;
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

  template <class T>
  void Matrix<T>::setup_matrix(int m, int n, ContextHandler::BlacsGrid &Grid)
  {
    nrows = m;
    ncols = n;
    initialise_descriptor(desc, Grid, local_nrows, local_ncols);
    // Allocate memory.
    store.resize(local_nrows*local_ncols);
  }

  template <class T>
  Matrix<T>::Matrix(const Matrix<T>& M) {
    nrows = M.nrows;
    ncols = M.ncols;
    block_nrows = M.block_ncols;
    block_ncols = M.block_nrows;
    local_nrows = M.local_ncols;
    local_ncols = M.local_nrows;
    lld = M.lld;
    izero = M.izero;
    init_row_idx = M.init_row_idx;
    init_col_idx = M.init_col_idx;
    info = M.info;
    store = M.store;
    desc = M.desc;
  }

  template <class T>
  void Matrix<T>::dump_data(H5::H5File &fh5, std::string group_name, std::string dataset_name)
  {
    H5::Exception::dontPrint();
    try {
      H5::Group group = fh5.openGroup(group_name.c_str());
    } catch (...) {
      H5::Group group = fh5.createGroup(group_name.c_str());
    }
    H5::Group group = fh5.openGroup(group_name.c_str());
    std::vector<hsize_t> dims(2);
    dims[0] = nrows;
    dims[1] = ncols;
    std::string dset_name = group_name + "/" + dataset_name;
    H5Helper::write(fh5, dset_name, store, dims);
  }

  // Destructor.
  template <class T>
  Matrix<T>::~Matrix()
  {
  }
}
#endif
