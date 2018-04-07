#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <time.h>
#include <mpi.h>
#include "json.hpp"
#include "kmeans.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpolatingPoints
{
  KMeans::KMeans(nlohmann::json &input)
  {
    filename = input.at("orbital_file").get<std::string>();
    max_it = input.at("kmeans").at("max_it").get<int>();
    threshold = input.at("kmeans").at("threshold").get<double>();
    thc_cfac = input.at("thc_cfac").get<int>();
    ndim = 3;
  }

  void KMeans::classify_grid_points(std::vector<double> &grid, std::vector<double> &centroids, std::vector<int> &grid_map, bool resize_deltas)
  {
    double dx, dy, dz;
    if (resize_deltas) deltas.resize(centroids.size()/ndim);
    for (int i = 0; i < grid.size()/ndim; i++) {
      // determine distance between input centroids and grid points.
      for (int j = 0; j < centroids.size()/ndim; j++) {
        // C order.
        dx = centroids[j*ndim] - grid[i*ndim];
        dy = centroids[j*ndim+1] - grid[i*ndim+1];
        dz = centroids[j*ndim+2] - grid[i*ndim+2];
        deltas[j] = sqrt(dx*dx + dy*dy + dz*dz);
      }
      // Assign grid point to centroid via argmin.
      grid_map[i] = std::distance(deltas.begin(), std::min_element(deltas.begin(), deltas.end()));
    }
  }

  void KMeans::update_centroids(std::vector<double> &rho, std::vector<double> &grid, std::vector<double> &centroids, std::vector<int> &grid_map)
  {
    for (int i = 0; i < num_interp_pts; i++) {
      weights[i] = 0; // class member / temporary storage.
      centroids[i*ndim] = 0;
      centroids[i*ndim+1] = 0;
      centroids[i*ndim+2] = 0;
    }
    for (int i = 0; i < num_grid_pts; i++) {
      weights[grid_map[i]] += rho[i];
      centroids[grid_map[i]*ndim] += rho[i] * grid[i*ndim];
      centroids[grid_map[i]*ndim+1] += rho[i] * grid[i*ndim+1];
      centroids[grid_map[i]*ndim+2] += rho[i] * grid[i*ndim+2];
    }
    for (int i = 0; i < num_interp_pts ; i++) {
      if (weights[i] < 1e-8) std::cout << "zero weight : " << i << " " << centroids[i] << " " << centroids[i+1] << " " << centroids[i+2] << std::endl;
      centroids[i*ndim] /= weights[i];
      centroids[i*ndim+1] /= weights[i];
      centroids[i*ndim+2] /= weights[i];
    }
  }

  std::vector<int> KMeans::map_to_grid(std::vector<double> &grid, std::vector<double> &centroids)
  {
    std::vector<int> interp_idxs(num_interp_pts);
    classify_grid_points(centroids, grid, interp_idxs, true);
    std::sort(interp_idxs.begin(), interp_idxs.end());
    for (int i = 0; i < interp_idxs.size()-1; i++) {
      if (interp_idxs[i] == interp_idxs[i+1]) {
        std::cout << "ERROR: Found repeated indices." << std::endl;
      }
    }
    return interp_idxs;
  }

  void KMeans::guess_initial_centroids(std::vector<double> &grid, std::vector<double> &centroids)
  {
    std::vector<int> tmp(num_grid_pts), indxs(num_interp_pts);
    for (int i = 0; i < grid.size()/ndim; i++) {
      tmp[i] = i;
    }
    std::mt19937 mt(7);
    std::shuffle(tmp.begin(), tmp.end(), mt);
    std::copy(tmp.begin(), tmp.begin()+indxs.size(), indxs.begin());
    std::sort(indxs.begin(), indxs.end());
    // fixed for ndim = 3.
    for (int i = 0; i < indxs.size(); i++) {
      centroids[i*ndim] = grid[indxs[i]*ndim];
      centroids[i*ndim+1] = grid[indxs[i]*ndim+1];
      centroids[i*ndim+2] = grid[indxs[i]*ndim+2];
    }
  }

  void KMeans::kernel(ContextHandler::BlacsHandler &BH, std::vector<int> &interp_indxs)
  {
    // real space grid.
    DistributedMatrix::Matrix<double> grid(filename, "real_space_grid", BH.Root);
    // "electron density" from supercell atomic orbitals.
    DistributedMatrix::Matrix<double> density(filename, "density", BH.Root);
    std::vector<hsize_t> dims(2);
    H5Helper::read_dims(filename, "aoR", dims);
    num_interp_pts = thc_cfac * dims[1];
    num_grid_pts = dims[0];
    deltas.resize(num_interp_pts);
    weights.resize(num_interp_pts);
    std::vector<double> current_centroids(num_interp_pts*ndim), new_centroids(num_interp_pts*ndim);
    std::vector<int> grid_map(num_grid_pts);
    interp_indxs.resize(num_interp_pts);
    bool root = BH.Root.rank == 0;

    double diff, t_kmeans = clock();
    if (root) {
      std::cout << "#################################################" << std::endl;
      std::cout << "## Finding interpolation points using K-Means. ##" << std::endl;
      std::cout << "#################################################" << std::endl;
      std::cout << std::endl;
      std::cout << " * Number of interpolating points : " << num_interp_pts << std::endl;
      std::cout << " * Number of grid points : " << num_grid_pts << std::endl;
      guess_initial_centroids(grid.store, current_centroids);
      for (int i = 0; i < max_it; i++) {
        classify_grid_points(grid.store, current_centroids, grid_map);
        update_centroids(density.store, grid.store, new_centroids, grid_map);
        diff = MatrixOperations::normed_difference(new_centroids, current_centroids);
        diff /= num_interp_pts;
        if (i % 10 == 0) std::cout << "  * Step: " << i << " Error: " << diff << std::endl;
        if (diff < threshold) {
          interp_indxs = map_to_grid(grid.store, new_centroids);
          break;
        } else {
          new_centroids.swap(current_centroids);
        }
      }
      if (diff > threshold) {
        std::cout << " * Threshold not breached. Final Error: " << diff << std::endl;
        new_centroids.swap(current_centroids);
        interp_indxs = map_to_grid(grid.store, new_centroids);
      }
    }
    if (root) {
      t_kmeans = clock() - t_kmeans;
      std::cout << " * Time taken for K-Means solution: " << t_kmeans / CLOCKS_PER_SEC << " seconds" << std::endl;
      std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  KMeans::~KMeans()
  {
  }
}
