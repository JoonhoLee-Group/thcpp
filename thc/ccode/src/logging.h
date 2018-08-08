#ifndef LOGGING_H
#define LOGGING_H
#include "distributed_matrix.h"
#include "matrix_operations.h"
#include "context_handler.h"

namespace Logging
{
  template <typename T>
  void dump_matrix(DistributedMatrix::Matrix<T> &M, std::string file_name, std::string matrix_name, bool row_major, bool root)
  {
    if (root) {
      MatrixOperations::local_transpose(M, row_major);
      MatrixOperations::swap_dims(M);
      std::cout << " * DEBUG: Writing " << matrix_name << " to file." << std::endl;
      H5::Exception::dontPrint();
      H5::H5File file = H5::H5File(file_name.c_str(), H5F_ACC_RDWR);
      try {
        H5::Group base = file.createGroup("/Hamiltonian");
      } catch (H5::FileIException) {
        H5::Group base = file.openGroup("/Hamiltonian");
      }
      M.dump_data(file, "/Hamiltonian/THC", matrix_name);
      MatrixOperations::local_transpose(M, !row_major);
      MatrixOperations::swap_dims(M);
    }
  }
}
#endif
