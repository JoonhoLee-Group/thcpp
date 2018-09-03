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
      QRCPSolver(nlohmann::json &input, int cfac, ContextHandler::BlacsHandler &BH);
      ~QRCPSolver();
      void kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs);
    private:
      // Variables
      std::string input_file;
      int filename_size;
      int thc_cfac;
  };
}
#endif
