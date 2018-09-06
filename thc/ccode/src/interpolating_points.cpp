#include <iostream>
#include <complex>
#include "json.hpp"
#include "context_handler.h"
#include "kmeans.h"
#include "qrcp.h"
#include "interpolating_points.h"

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
      nlohmann::json base = input.at("interpolating_points").at("solver");
      try {
        nlohmann::json kmeans = base.at("kmeans");
        kmeans = true;
      }
      catch (nlohmann::json::out_of_range& error) {
        qrcp = true;
        try {
          nlohmann::json qrcp = base.at("qrcp");
        }
        catch (nlohmann::json::out_of_range& error) {
          std::cout << "No solver specified." << std::endl;
        }
      }
    }
  }
  std::vector<int> IPoints::kernel(nlohmann::json &input, ContextHandler::BlacsHandler &BH, int thc_cfac, bool half_rotate)
  {
    std::vector<int> interp_indxs;
    if (qrcp) {
      QRCP::QRCPSolver QR(input, BH);
      QR.kernel(BH, interp_indxs, thc_cfac, half_rotate);
    } else if (kmeans) {
      KMeans::KMeansSolver KM(input, BH);
      KM.kernel(BH, interp_indxs, thc_cfac);
    } else {
      std::cout << "No interpolating point solver specified." << std::endl;
    }
    return interp_indxs;
  }
}
