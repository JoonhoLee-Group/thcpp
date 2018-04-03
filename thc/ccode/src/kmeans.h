#ifndef KMEANS_H
#define KMEANS_H
#include <iostream>
#include <vector>
#include <random>
#include "context_handler.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpolatingPoints
{
  class KMeans
  {
    public:
      KMeans(std::string input_data, int max_iteration, double thresh, int cfac);
      ~KMeans();
      void kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs, DistributedMatrix::Matrix<double> &aoR);
    private:
      void classify_grid_points(std::vector<double> &grid, std::vector<double> &centroids,
                                std::vector<int> &grid_map, bool resize_deltas=false);
      void update_centroids(std::vector<double> &rho, std::vector<double> &grid,
                            std::vector<double> &centroids, std::vector<int> &grid_map);
      std::vector<int> map_to_grid(std::vector<double> &grid, std::vector<double> &centroids);
      void guess_initial_centroids(std::vector<double> &grid, std::vector<double> &centroids);
      // Variables
      std::string filename;
      int max_it;
      double threshold;
      int num_interp_pts, num_grid_pts;
      int thc_cfac;
      int ndim;
      // temporary storage.
      std::vector<double> deltas, weights;
  };
}
#endif
