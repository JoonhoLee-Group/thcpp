#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "kmeans.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpKMeans
{
  KMeans::KMeans(std::string filename, int max_it, int thresh, int cfac)
  {
    filename = filename;
    max_it = max_it;
    threshold = thresh;
    thc_cfac = cfac;
  }

  void KMeans::classify_grid_points(std::vector<double> &grid, std::vector<double> &centroids, std::vector<int> &grid_map, bool resize_deltas)
  {
    double dx, dy, dz;
    if (resize_deltas) deltas.resize(centroids.size());
    for (int i = 0; i < grid_map.size(); i++) {
      // determine distance between input centroids and grid points.
      for (int j = 0; j < centroids.size(); j++) {
        // C order.
        dx = centroids[j]-grid[i];
        dy = centroids[j+1]-grid[i+1];
        dz = centroids[j+2]-grid[i+2];
        deltas[j] = sqrt(dx*dx + dy*dy + dz*dz);
      }
      // Assign grid point to centroid via argmin.
      grid_map[i] = std::distance(deltas.begin(), std::min_element(deltas.begin(), deltas.end()));
    }
  }

  void KMeans::update_centroids(std::vector<double> &rho, std::vector<double> &grid, std::vector<double> &centroids, std::vector<int> &grid_map)
  {
    for (int i = 0; i < centroids.size(); i++) {
      weights[i] = 0; // class member / temporary storage.
      centroids[i] = 0;
    }
    for (int i = 0; i < rho.size(); i++) {
      weights[grid_map[i]] += rho[i];
      centroids[grid_map[i]] += rho[i] * grid[i];
      centroids[grid_map[i]+1] += rho[i] * grid[i+1];
      centroids[grid_map[i]+2] += rho[i] * grid[i+2];
    }
    for (int i = 0; i < centroids.size(); i++) {
      centroids[i] /= weights[i];
    }
  }

  std::vector<int> KMeans::map_to_grid(std::vector<double> &grid, std::vector<double> &centroids)
  {
    std::vector<int> interp_idxs(centroids.size());
    classify_grid_points(centroids, grid, interp_idxs, true);
    std::sort(interp_idxs.begin(), interp_idxs.end());
    for (int i = 0; i < interp_idxs.size()-1; i++) {
      if (interp_idxs[i] = interp_idxs[i+1]) {
        std::cout << "ERROR: Found repeated indices." << std::endl;
      }
    }
    return interp_idxs;
  }

  void KMeans::guess_initial_centroids(std::vector<double> &grid, std::vector<double> &centroids)
  {
    std::vector<int> tmp(grid.size()), indxs(centroids.size());
    for (int i = 0; i < grid.size(); i++) {
      tmp[i] = i;
    }
    std::mt19937 mt(7);
    std::shuffle(tmp.begin(), tmp.end(), mt);
    std::sort(tmp.begin(), tmp.end());
    std::copy(tmp.begin(), tmp.begin()+indxs.size(), indxs.begin());
    for (int i = 0; i < indxs.size(); i++) {
      centroids[i] = grid[indxs[i]];
    }
  }

  std::vector<int> KMeans::kernel(ContextHandler::BlacsHandler &BH)
  {
    // real space supercell atomic orbitals.
    DistributedMatrix::Matrix aoR(filename, "aoR", BH.Root);
    // real space grid.
    DistributedMatrix::Matrix grid(filename, "real_space_grid", BH.Root);
    // "electron density" from supercell atomic orbitals.
    DistributedMatrix::Matrix density(filename, "density", BH.Root);
    num_interp_pts = thc_cfac * aoR.ncols;
    num_grid_pts = aoR.nrows;
    deltas.resize(num_interp_pts);
    weights.resize(num_interp_pts);
    std::vector<double> current_centroids(num_interp_pts), new_centroids(num_interp_pts);
    std::vector<int> grid_map(num_grid_pts);
    guess_initial_centroids(grid.store, current_centroids);

    double diff;
    for (int i = 0; i < max_it; i++) {
      classify_grid_points(grid.store, current_centroids, grid_map);
      update_centroids(density.store, grid.store, new_centroids, grid_map);
      diff = MatrixOperations::normed_difference(new_centroids, current_centroids);
      if (diff < threshold) {
        return map_to_grid(grid.store, new_centroids);
      } else {
        std::copy(new_centroids.begin(), new_centroids.end(), current_centroids.begin());
      }
    }
    std::cout << "Threshold not breached: " << diff << std::endl;
  }
}
