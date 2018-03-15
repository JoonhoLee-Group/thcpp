#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H 
#include <iostream>
#include <vector>
#include "mkl.h"
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

inline std::vector<double> least_squares(std::vector<double> &A, std::vector<double> &B, int nrow, int ncol, int nrhs)
{
  const char transa = 'N';
  std::vector<double> AT(nrow*ncol), BT(nrow*nrhs), SOL(nrow*nrhs), WORK(1);
  int lwork = -1, info;
  // Transpose A and B matrices for fortran formatting.
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      AT[j*nrow+i] = A[i*ncol+j]; 
    }
  }
  std::cout << "AT " << std::endl;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < nrhs; j++) {
      BT[j*nrow+i] = B[i*nrhs+j]; 
    }
  }
  std::cout << "BT " << std::endl;
  const char trans = 'N';
  dgels(&trans, &nrow, &ncol, &nrhs, AT.data(), &nrow, BT.data(), &nrhs, WORK.data(), &lwork, &info);
  std::cout << "DGELS " <<  " " << B.size() << " " << nrow*nrhs << std::endl;
  // Transpose array to row-major format.
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < nrhs; j++) {
      SOL[i*nrhs+j] = BT[j*nrow+i];
    }
  }
  std::cout << "SOL" << "  " << lwork << " " << info << " " << vector_sum(SOL) << " " << vector_sum(BT) << std::endl;
  return SOL;
}
inline void least_squares_lapacke(std::vector<double> &A, std::vector<double> &B, int nrow, int ncol, int nrhs)
{
  const char transa = 'N';
  std::vector<double> WORK(1);
  int lwork = -1, info;
  // Transpose A and B matrices for fortran formatting.
  info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', nrow, ncol, nrhs, A.data(), nrow, B.data(), nrhs);
}

}
#endif
