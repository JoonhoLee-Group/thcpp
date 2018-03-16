#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H 
#include <iostream>
#include <vector>
#include "mkl.h"
#include "mkl_lapack.h"
#include "mkl_lapacke.h"

namespace MatrixOperations
{
template <typename T>
T vector_sum(std::vector<T> vec)
{
  T sum = 0;
  for (int i = 0; i < vec.size(); i++) {
    sum += vec[i];
  }
  return sum;
}

template <typename T>
inline void product(int M, int N, int K, T one,
                    T* A, int LDA, T* B, int LDB,
                    T zero, T* C, int LDC)
{
  const char transa = 'N';
  const char transb = 'N';

  // C = A*B -> fortran -> C' = B'*A', 
  dgemm(&transa, &transb, &N, &M, &K, &one, B, &LDB, A, &LDA, &zero, C, &LDC);  
}

// Using native fortran interface to lapack. Assumes arrays are in column major format.
inline void least_squares(double *A, double *B, int nrow, int ncol, int nrhs)
{
  const char trans = 'N';
  int lwork = -1, info;
  std::vector<double> WORK(1);
  // Workspace query.
  dgels_(&trans, &nrow, &ncol, &nrhs, A, &nrow, B, &nrow, WORK.data(), &lwork, &info);
  lwork = WORK[0];
  WORK.resize(lwork);
  // Actually perform least squares.
  dgels_(&trans, &nrow, &ncol, &nrhs, A, &nrow, B, &nrow, WORK.data(), &lwork, &info);
}

// Testing C interface to lapack. Assumes arrays are stored in row major format.
inline void least_squares_lapacke(std::vector<double> &A, std::vector<double> &B, int nrow, int ncol, int nrhs)
{
  const char transa = 'N';
  std::vector<double> WORK(1);
  int lwork = -1, info;
  info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', nrow, ncol, nrhs, A.data(), ncol, B.data(), nrhs);
}

}
#endif
