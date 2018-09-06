#include <iostream>
#include <complex>
#include "json.hpp"
#include "context_handler.h"
#include "kmeans.h"
#include "qrcp.h"

namespace InterpolatingPoints
{
  IPoints::IPoints(nlohmann::json &input, ContextHandler::BlacsHandler &BH)
  {
    int filename_size;
    if (BH.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << "##   Setting up interpolating point solver.    ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      nhlohmann::json base = input.at("interpolating_points").at("solver");
      try {
        nhlohmann::json kmeans = input.at("kmeans");
        kmeans = true;
      }
      catch (nlohmann::json::out_of_ranger& error) {
        qrcp = true;
        try {
          solver = input.at("qrcp")
        }
        catch (nlohmann::json::out_of_range& error) {
          std::cout << "No solver specified." << std::endl;
        }
      }
    }
  }
  std::vector<int> IPoints::kernel(ContextHandler::BlacsHandler &BH, int thc_cfac, bool half_rotate)
  {
    std::vector<int> interp_indxs;
    if (qrcp) {
      QRCP::QRCPSolver QR(input_options, BH);
      QR.kernel(BH, interp_indxs, thc_cfac, half_rotate);
    } else if (kmeans) {
      KMeans::KMeansSolver KM(input_options, BH);
      KM.kernel(BH, interp_indxs, thc_cfac);
    } else {
      std::cout << "No interpolating point solver specified." << std::endl;
    }
    return interp_indxs;
  }
}
