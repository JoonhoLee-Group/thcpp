#ifndef INTERPOLATING_VECTORS_H
#define INTERPOLATING_VECTORS_H
#include <iostream>
#include <complex>
#include "json.hpp"
#include "context_handler.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpolatingVectors
{
  class IVecs
  {
    public:
      IVecs(nlohmann::json &input_file, ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs);
      void kernel(ContextHandler::BlacsHandler &BH);
    private:
      void fft_vectors(ContextHandler::BlacsHandler &BH, DistributedMatrix::Matrix<std::complex<double> > &IVG);
      void dump_thc_data(DistributedMatrix::Matrix<std::complex<double> > &IVG, ContextHandler::BlacsHandler &BH);
      DistributedMatrix::Matrix<std::complex<double> > CCt, CZt;
      std::string input_file, output_file;
      int thc_cfac;
  };
}
#endif
