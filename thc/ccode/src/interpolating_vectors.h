#ifndef INTERPOLATING_VECTORS_H
#define INTERPOLATING_VECTORS_H
#include <iostream>
#include <complex>
#include "context_handler.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpolatingVectors
{
  class IVecs
  {
    public:
      IVecs(std::string file, ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs, DistributedMatrix::Matrix<double> &aoR);
      void kernel(ContextHandler::BlacsHandler &BH);
    private:
      void fft_vectors(ContextHandler::BlacsHandler &BH, DistributedMatrix::Matrix<std::complex<double> > &IVG);
      void dump_thc_data(std::string outfile, DistributedMatrix::Matrix<std::complex<double> > &IVG, ContextHandler::BlacsHandler &BH);
      DistributedMatrix::Matrix<double> CCt, CZt;
      std::string filename, outfile;
  };
}
#endif
