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
    int solver_size;
    if (BH.rank == 0) {
      std::cout << "#################################################" << std::endl;
      std::cout << "##   Setting up interpolating point solver.    ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      nlohmann::json base = input.at("interpolating_points");
      try {
        nlohmann::json kmeans_dict = base.at("kmeans");
        solver = "kmeans";
      }
      catch (nlohmann::json::out_of_range& error) {
        try {
          nlohmann::json qrcp = base.at("qrcp");
          solver = "qrcp";
        }
        catch (nlohmann::json::out_of_range& error) {
          std::cout << "No solver specified." << std::endl;
          solver = "kmeans";
        }
      }
      solver_size = solver.size();
    }
    MPI_Bcast(&solver_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (BH.rank != 0) solver.resize(solver_size);
    MPI_Bcast(&solver[0], solver.size()+1, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
  std::vector<int> IPoints::kernel(nlohmann::json &input, ContextHandler::BlacsHandler &BH, int thc_cfac, bool half_rotate)
  {
    std::vector<int> interp_indxs;
    if (solver == "qrcp") {
      if (BH.rank == 0) {
        std::cout << " * Using QRCP solver." << std::endl;
        std::cout << std::endl;
      }
      QRCP::QRCPSolver QR(input, BH);
      QR.kernel(BH, interp_indxs, thc_cfac, half_rotate);
    } else if (solver == "kmeans") {
      if (BH.rank == 0) {
        std::cout << " * Using K-Means solver." << std::endl;
        std::cout << std::endl;
      }
      KMeans::KMeansSolver KM(input, BH);
      KM.kernel(BH, interp_indxs, thc_cfac);
    } else {
      if (BH.rank == 0) {
        std::cout << " * No interpolating point solver specified." << std::endl;
      }
    }
    return interp_indxs;
  }
}
