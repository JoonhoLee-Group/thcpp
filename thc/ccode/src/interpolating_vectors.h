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
      IVecs(nlohmann::json &input_file, ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs, bool half_rotated, bool append);
      void kernel(ContextHandler::BlacsHandler &BH);
      void dump_qmcpack_data(int thc_cfac, int thc_half_cfac,
                             ContextHandler::BlacsHandler &BH);
    private:
      void fft_vectors(ContextHandler::BlacsHandler &BH, DistributedMatrix::Matrix<std::complex<double> > &IVG,
                       DistributedMatrix::Matrix<std::complex<double> > &IVMG);
      void construct_muv(DistributedMatrix::Matrix<std::complex<double> > &IVG,
                         DistributedMatrix::Matrix<std::complex<double> > &IVMG,
                         DistributedMatrix::Matrix<std::complex<double> > &Muv,
                         ContextHandler::BlacsHandler &BH);
      void dump_thc_data(DistributedMatrix::Matrix<std::complex<double> > &Muv,
                         ContextHandler::BlacsHandler &BH);
      void setup_CZt(std::vector<int> &interp_indxs, ContextHandler::BlacsHandler &BH);
      void setup_CZt_half(std::vector<int> &interp_indxs, ContextHandler::BlacsHandler &BH);
      void setup_CCt(std::vector<int> &interp_indxs, ContextHandler::BlacsHandler &BH);
      void check_rank(ContextHandler::BlacsHandler &BH);
      void setup_pseudo_dm(DistributedMatrix::Matrix<std::complex<double> > &Pua, std::vector<int> &interp_indxs,
                           ContextHandler::BlacsHandler &BH, std::string aos, bool write, std::string prfx);
      DistributedMatrix::Matrix<std::complex<double> > CCt, CZt;
      std::string input_file, output_file;
      int thc_cfac, thc_half_cfac;
      int nbasis;
      bool half_rotate;
      std::string prefix;
  };
}
#endif
