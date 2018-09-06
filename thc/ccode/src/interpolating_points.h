#ifndef INTERPOLATING_POINTS_H
#define INTERPOLATING_POINTS_H
#include <iostream>
#include <complex>
#include "json.hpp"
#include "context_handler.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpolatingPoints
{
  class IPoints
  {
    public:
      IPoints(nlohmann::json &input_file, ContextHandler::BlacsHandler &BH);
      std::vector<int> kernel(nlohmann::json &input_file, ContextHandler::BlacsHandler &BH, int thc_cfac, bool half_rotate=false);

    private:
      bool kmeans = false, qrcp = false;
  };
}
#endif
