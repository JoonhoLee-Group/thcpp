#ifndef QRCP_H
#define QRCP_H
#include <iostream>
#include <vector>
#include <random>
#include "json.hpp"
#include "context_handler.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace QRCP
{
  class QRCPSolver
  {
    public:
      QRCPSolver(nlohmann::json &input, ContextHandler::BlacsHandler &BH);
      ~QRCPSolver();
      void kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs,
                  int thc_cfac, bool half_rotate=false);
    private:
      // Variables
      std::vector<int> map_perm(std::vector<int> &perm, int ncols, int local_ncols, ContextHandler::BlacsHandler &BH);
      std::string input_file;
      int filename_size;
      int thc_cfac;
      // Sub sample rows of Z matrix using random gaussian matrices.
      bool sub_sample;
  };
}
#endif
