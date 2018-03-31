#ifndef CONTEXT_HANDLER_H
#define CONTEXT_HANDLER_H
#include "cblacs_defs.h"

namespace ContextHandler
{
  class BlacsGrid
  {
      public:
        BlacsGrid() {};
        ~BlacsGrid();
        BlacsGrid(int nr, int nc);
        int nrows, ncols;
        int row, col;
        int ctxt;
  };
  // Handle different blacs grid distributions
  class BlacsHandler
  {
    public:
      BlacsHandler();
      ~BlacsHandler();
      int rank;
      int nprocs, proc_rows, proc_cols;
      BlacsGrid Root;
      BlacsGrid Square;
      BlacsGrid Column;
  };
}
#endif
