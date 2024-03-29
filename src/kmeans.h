#ifndef KMEANS_H
#define KMEANS_H
#include <iostream>
#include <vector>
#include <random>
#include "json.hpp"
#include "context_handler.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace KMeans
{
  class KMeansSolver
  {
    public:
      KMeansSolver(nlohmann::json &input, ContextHandler::BlacsHandler &BH);
      ~KMeansSolver();
      void kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs, int thc_cfac);
    private:
      void classify_grid_points(std::vector<double> &grid, std::vector<double> &centroids,
                                std::vector<int> &grid_map, bool resize_deltas=false);
      void update_centroids(std::vector<double> &rho, std::vector<double> &grid,
                            std::vector<double> &centroids, std::vector<int> &grid_map);
      std::vector<int> map_to_grid(std::vector<double> &grid, std::vector<double> &centroids);
      void guess_initial_centroids(std::vector<double> &grid, std::vector<double> &centroids);
      void scatter_data(std::vector<double> &grid, int num_points, int ndim, ContextHandler::BlacsGrid &BG);
      void gather_data(std::vector<double> &grid, ContextHandler::BlacsGrid &BG);
      double percentage_change(std::vector<int> &new_grid_map, std::vector<int> &old_grid_map);
      // Variables
      std::string filename;
      int max_it;
      double threshold;
      int num_interp_pts, num_grid_pts;
      int thc_cfac;
      int ndim;
      unsigned int rng_seed;
      std::string density_label;
      // temporary storage.
      std::vector<double> deltas, weights, global_weights, global_centroids;
  };
}
#endif
