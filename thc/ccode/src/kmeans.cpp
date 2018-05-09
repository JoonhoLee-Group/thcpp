#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <time.h>
#include <chrono>
#include <mpi.h>
#include "json.hpp"
#include "kmeans.h"
#include "distributed_matrix.h"
#include "matrix_operations.h"

namespace InterpolatingPoints
{
  KMeans::KMeans(nlohmann::json &input, int cfac, ContextHandler::BlacsHandler &BH)
  {
    if (BH.rank == 0) {
      filename = input.at("orbital_file").get<std::string>();
      max_it = input.at("kmeans").at("max_it").get<int>();
      threshold = input.at("kmeans").at("threshold").get<double>();
      try {
        rng_seed = input.at("kmeans").at("rng_seed").get<int>();
      }
      catch (nlohmann::json::out_of_range& error) {
        std::cout << " * RNG seed not set in input file." << std::endl;
        rng_seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::cout << " * Setting RNG seed to : " << rng_seed << std::endl;
        std::cout << std::endl;
      }
    }
    MPI_Bcast(&max_it, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    thc_cfac = cfac;
    ndim = 3;
  }

  void KMeans::scatter_data(std::vector<double> &grid, int num_points, int ndim, ContextHandler::BlacsGrid &BG)
  {
    int num_points_per_proc = num_points / BG.nprocs;
    std::vector<int> send_counts(BG.nprocs), disps(BG.nprocs);
    int nleft = num_points;
    if (BG.rank == 0) {
      for (int i = 0; i < BG.nprocs-1; i++) {
        send_counts[i] = num_points_per_proc*ndim;
        nleft -= num_points_per_proc;
      }
      send_counts[BG.nprocs-1] = nleft*ndim;
      disps[0] = 0;
      for (int i = 1; i < BG.nprocs; i++) {
        disps[i] = disps[i-1] + send_counts[i-1];
      }
    }
    int nrecv;
    MPI_Scatter(send_counts.data(), 1, MPI_INT, &nrecv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> recv_buf(nrecv);
    MPI_Scatterv(grid.data(), send_counts.data(), disps.data(), MPI_DOUBLE,
                 recv_buf.data(), recv_buf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    grid.swap(recv_buf);
  }

  void KMeans::gather_data(std::vector<double> &grid, ContextHandler::BlacsGrid &BG)
  {
    std::vector<int> recv_counts(BG.nprocs), disps(BG.nprocs);
    int nsend = grid.size();
    MPI_Gather(&nsend, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> recv_buf;
    if (BG.rank == 0) {
      disps[0] = 0;
      for (int i = 1; i < BG.nprocs; i++) {
        disps[i] = disps[i-1] + recv_counts[i-1];
      }
      recv_buf.resize(MatrixOperations::vector_sum(recv_counts));
    }
    int ierr = MPI_Gatherv(grid.data(), nsend, MPI_DOUBLE,
                           recv_buf.data(), recv_counts.data(), disps.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    grid.swap(recv_buf);
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
      grid_map[i] = std::distance(deltas.begin(),
                                  std::min_element(deltas.begin(),
                                                   deltas.end()));
    }
  }

  void KMeans::update_centroids(std::vector<double> &rho, std::vector<double> &grid, std::vector<double> &centroids, std::vector<int> &grid_map)
  {
    MatrixOperations::zero(weights);
    MatrixOperations::zero(centroids);
    MatrixOperations::zero(global_weights);
    MatrixOperations::zero(global_centroids);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (int i = 0; i < grid.size()/ndim; i++) {
      weights[grid_map[i]] += rho[i];
      centroids[grid_map[i]*ndim] += rho[i] * grid[i*ndim];
      centroids[grid_map[i]*ndim+1] += rho[i] * grid[i*ndim+1];
      centroids[grid_map[i]*ndim+2] += rho[i] * grid[i*ndim+2];
    }
    MPI_Allreduce(centroids.data(), global_centroids.data(), centroids.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    centroids.swap(global_centroids);
    MPI_Allreduce(weights.data(), global_weights.data(), weights.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    weights.swap(global_weights);
    for (int i = 0; i < num_interp_pts; i++) {
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
        std::cout << "ERROR: Found repeated indices: " << i << std::endl;
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
    std::mt19937 mt(rng_seed);
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
    if (BH.rank == 0) H5Helper::read_dims(filename, "aoR", dims);
    MPI_Bcast(dims.data(), dims.size(), MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    num_interp_pts = thc_cfac * dims[1];
    num_grid_pts = dims[0];
    // Store for computing argmin.
    deltas.resize(num_interp_pts);
    // Global array for computing centroid weights.
    weights.resize(num_interp_pts);
    // Global array for MPI reduction.
    global_weights.resize(num_interp_pts);
    // Stores for computed centroids.
    std::vector<double> current_centroids(num_interp_pts*ndim), new_centroids(num_interp_pts*ndim), tmp_cntr(num_interp_pts*ndim);
    // Global array for MPI reduction.
    global_centroids.resize(num_interp_pts*ndim);
    // Store for interpolating indices.
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
    }
    MPI_Bcast(current_centroids.data(), current_centroids.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Since the grid is stored in C order it's easier to just redistribute it by hand
    // rather than using SCALAPACK. SCALAPACK work for data stored contiguously in row
    // major order and I don't want to transpose things.
    scatter_data(grid.store, grid.nrows, ndim, BH.Column);
    scatter_data(density.store, grid.nrows, 1, BH.Column);
    // Maps grid point to centroid.
    std::vector<int> grid_map(density.store.size());
    for (int i = 0; i < max_it; i++) {
      classify_grid_points(grid.store, current_centroids, grid_map);
      update_centroids(density.store, grid.store, new_centroids, grid_map);
      diff = MatrixOperations::normed_difference(new_centroids, current_centroids);
      if (i % 10 == 0 && BH.rank == 0) std::cout << "  * Step: " << i << " Error: " << diff << std::endl;
      if (diff < threshold) {
        gather_data(grid.store, BH.Column);
        if (BH.rank == 0) {
          interp_indxs = map_to_grid(grid.store, new_centroids);
        }
        break;
      } else {
        new_centroids.swap(current_centroids);
      }
    }
    if (diff > threshold) {
      std::cout << " * Threshold not breached. Final Error: " << diff << std::endl;
      new_centroids.swap(current_centroids);
      gather_data(grid.store, BH.Column);
      if (BH.rank == 0) {
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
