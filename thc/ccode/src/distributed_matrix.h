#ifndef DISTRIBUTED_MATRIX_H
#define DISTRIBUTED_MATRIX_H
#include <iostream>
#include <vector>
#include <complex>
#include "context_handler.h"

// Operations for matrices distributed using scalapack.
// Asumes column major (Fortran) order for layout.
// Thin wrapper to distribute data to and from block cyclic form.
namespace DistributedMatrix
{
  class Matrix
  {
    public:
      Matrix(int nrows, int ncols, ContextHandler::BlacsGrid &Grid);
      Matrix(std::string filename, std::string name,
             ContextHandler::BlacsGrid &Grid, bool row_major=true);
      ~Matrix();
      void gather_block_cyclic(int ctxt);
      void gather_fft(int ctxt);
      void scatter_block_cyclic(int ctxt);
      void redistribute(ContextHandler::BlacsGrid &GridA, ContextHandler::BlacsGrid &GridB);
      void redistribute_to_column_cyclic(int ctxt);
      void initialise_discriptors(int ctxt, int root_ctxt, int ccyc_ctxt);
      void initialise_discriptor(std::vector<int> &desc, ContextHandler::BlacsGrid &Grid, int &nr, int &nc);
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
      std::vector<double> store;
      // scalapack descriptor array for current blacs context.
      std::vector<int> desc;
  };
}
#endif
