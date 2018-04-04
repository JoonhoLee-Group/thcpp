#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H
#include <iostream>
#include <vector>
#include <cmath>
#include "lapack_defs.h"
#include "scalapack_defs.h"
#include "distributed_matrix.h"

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

inline double normed_difference(std::vector<double> &a, std::vector<double> &b)
{
  double diff = 0.0;
  for (int i = 0; i < a.size(); i++) {
    diff += (a[i]-b[i])*(a[i]-b[i]);
  }
  return sqrt(diff);
}

template <typename T>
inline void product(int M, int N, int K, T one,
                    T* A, int LDA, T* B, int LDB,
                    T zero, T* C, int LDC)
{
  const char transa = 'N';
  const char transb = 'N';

  // C = A*B -> fortran -> C' = B'*A',
  dgemm_(&transa, &transb, &N, &M, &K, &one, B, &LDB, A, &LDA, &zero, C, &LDC);
}

// distributed matrix product
inline void product(DistributedMatrix::Matrix<double> &A, DistributedMatrix::Matrix<double> &B,
                    DistributedMatrix::Matrix<double> &C)
{
  char transa = 'N', transb = 'N';
  double one = 1.0, zero = 0.0;
  pdgemm_(&transa, &transb, &A.nrows, &B.ncols, &A.ncols,
          &one,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          &zero,
          C.store.data(), &C.init_row_idx, &C.init_col_idx, C.desc.data());
}

// distributed matrix least squares solve.
inline void least_squares(DistributedMatrix::Matrix<double> &A, DistributedMatrix::Matrix<double> &B)
{
  char trans = 'N';
  int lwork = -1, info;
  std::vector<double> WORK(1);
  // Workspace query.
  pdgels_(&trans, &A.nrows, &A.ncols, &B.ncols,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          WORK.data(), &lwork, &info);
  lwork = WORK[0];
  WORK.resize(lwork);
  // Actually perform least squares.
  pdgels_(&trans, &A.nrows, &A.ncols, &B.ncols,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          WORK.data(), &lwork, &info);
}

// Using native fortran interface to lapack. Assumes arrays are in column major format.
inline void least_squares(double *A, double *B, int nrow, int ncol, int nrhs)
{
  char trans = 'N';
  int lwork = -1, info;
  std::vector<double> WORK(1);
  // Workspace query.
  dgels_(&trans, &nrow, &ncol, &nrhs, A, &nrow, B, &nrow, WORK.data(), &lwork, &info);
  lwork = WORK[0];
  WORK.resize(lwork);
  // Actually perform least squares.
  dgels_(&trans, &nrow, &ncol, &nrhs, A, &nrow, B, &nrow, WORK.data(), &lwork, &info);
}

template <typename T>
inline void down_sample(DistributedMatrix::Matrix<T> &A, DistributedMatrix::Matrix<T> &B, std::vector<int> &indices, int offset)
{
    for (int i = 0; i < indices.size(); ++i) {
      int ix = indices[i];
      std::copy(A.store.begin()+ix*offset,
                A.store.begin()+(ix+1)*offset,
                B.store.begin()+i*offset);
    }
}

inline void transpose(DistributedMatrix::Matrix<double> &A, DistributedMatrix::Matrix<double> &AT)
{
  char trans = 'T';
  double one = 1.0, zero = 0.0;
  pdgeadd_(&trans, &A.ncols, &A.nrows,
           &one,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           &zero,
           AT.store.data(), &AT.init_row_idx, &AT.init_col_idx, AT.desc.data());
}

// Testing C interface to lapack. Assumes arrays are stored in row major format.
//inline void least_squares_lapacke(std::vector<double> &A, std::vector<double> &B, int nrow, int ncol, int nrhs)
//{
  //int info;
  //info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', nrow, ncol, nrhs, A.data(), ncol, B.data(), nrhs);
//}
//

inline void redistribute(int m, int n,
                         double *a, int ia, int ja, int *desca,
                         double *b, int ib, int jb, int *descb,
                         int ictxt)
{
    pdgemr2d_(&m, &n, a, &ia, &ib, desca, b, &ia, &ib, descb, &ictxt);
}

inline void redistribute(int m, int n,
                         std::complex<double> *a, int ia, int ja, int *desca,
                         std::complex<double> *b, int ib, int jb, int *descb,
                         int ictxt)
{
    pzgemr2d_(&m, &n, a, &ia, &ib, desca, b, &ia, &ib, descb, &ictxt);
}

template <typename T>
inline void redistribute(DistributedMatrix::Matrix<T> &M, ContextHandler::BlacsGrid &GridA, ContextHandler::BlacsGrid &GridB, bool verbose=false, int block_rows=-1, int block_cols=-1)
{
  // setup descriptor for Blacs grid we'll distribute to.
  std::vector<int> descb(9);
  if (block_rows > 0 and block_cols > 0) {
    M.initialise_descriptor(descb, GridB, M.local_nrows, M.local_ncols, block_rows, block_cols);
  } else {
    M.initialise_descriptor(descb, GridB, M.local_nrows, M.local_ncols);
  }
  std::vector<T> tmp(M.local_nrows*M.local_ncols);
  int ctxt;
  // ctxt for p?gemr2d call must at least contain the union of processors from gridA and
  // gridB.
  if (GridA.nrows*GridA.ncols >= GridB.nrows*GridB.ncols) {
    ctxt = GridA.ctxt;
  } else {
    ctxt = GridB.ctxt;
  }
  redistribute(M.nrows, M.ncols,
               M.store.data(), M.init_row_idx, M.init_col_idx, M.desc.data(),
               tmp.data(), M.init_row_idx, M.init_col_idx, descb.data(),
               ctxt);
  tmp.swap(M.store);
  descb.swap(M.desc);
  if (GridB.rank == 0 && verbose) {
    double memory = UTILS::get_memory(M.store);
    std::cout << "  * Local memory usage (on root processor) following redistribution: " << memory << " GB" << std::endl;
    std::cout << "  * Local shape (on root processor) following redistribution: (" << M.local_nrows << ", " << M.local_ncols << ")" << std::endl;
  }
}

}
#endif
