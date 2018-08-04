#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H
#include <iostream>
#include <vector>
#include <algorithm>
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

template <typename T>
void zero(std::vector<T> &vec)
{
  T sum = 0;
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = 0;
  }
}

inline double normed_difference(std::vector<double> &a, std::vector<double> &b)
{
  double diff = 0.0;
  double max_a = *std::max_element(a.begin(), a.end());
  double min_a = *std::min_element(a.begin(), a.end());
  double abs_max_a = std::max(std::abs(max_a), std::abs(min_a));
  double max_b = *std::max_element(b.begin(), b.end());
  double min_b = *std::min_element(b.begin(), b.end());
  double abs_max_b = std::max(std::abs(max_b), std::abs(min_b));
  // Triangle inequality.
  double scale = abs_max_a + abs_max_b;
  for (int i = 0; i < a.size(); i++) {
    diff += ((a[i]-b[i])/scale)*((a[i]-b[i])/scale);
  }
  return scale * sqrt(diff);
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

inline void product(DistributedMatrix::Matrix<std::complex<double> > &A, DistributedMatrix::Matrix<std::complex<double> > &B,
                    DistributedMatrix::Matrix<std::complex<double> > &C, char transA='N', char transB='N')
{
  std::complex<double>  one = 1.0, zero = 0.0;
  int m, n, k;
  if ((transA == 'T' || transA == 'C') && transB == 'N') {
    m = A.ncols;
    n = B.ncols;
    k = A.nrows;
  } else if ((transB == 'T' || transB == 'C') && transA == 'N') {
    m = A.nrows;
    n = B.nrows;
    k = A.ncols;
  } else if ((transA == 'T' || transA == 'C') && (transB == 'T' || transB == 'C')) {
    m = A.ncols;
    n = A.nrows;
    k = A.nrows;
  } else {
    m = A.nrows;
    n = A.ncols;
    k = A.ncols;
  }
  pzgemm_(&transA, &transB, &m, &n, &k,
          &one,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          &zero,
          C.store.data(), &C.init_row_idx, &C.init_col_idx, C.desc.data());
}

// distributed matrix least squares solve.
inline int least_squares(DistributedMatrix::Matrix<double> &A, DistributedMatrix::Matrix<double> &B)
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
  return info;
}

inline int least_squares(DistributedMatrix::Matrix<std::complex<double> > &A, DistributedMatrix::Matrix<std::complex<double> > &B)
{
  char trans = 'N';
  int lwork = -1, info;
  std::vector<std::complex<double> > WORK(1);
  // Workspace query.
  pzgels_(&trans, &A.nrows, &A.ncols, &B.ncols,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          WORK.data(), &lwork, &info);
  lwork = int(WORK[0].real());
  WORK.resize(lwork);
  // Actually perform least squares.
  pzgels_(&trans, &A.nrows, &A.ncols, &B.ncols,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          WORK.data(), &lwork, &info);
  return info;
}

inline int least_squares(std::complex<double> *A, std::complex<double> *B, int nrow, int ncol, int nrhs, int &rank, std::vector<double> &s)
{
  int lwork = -1, info;
  std::vector<std::complex<double> > WORK(1);
  std::vector<double> RWORK(5*std::min(nrow,ncol));
  // machine precision
  double rcond = 1e-16*std::max(nrow,ncol);
  // Workspace query.
  zgelss_(&nrow, &ncol, &nrhs, A, &nrow, B, &nrow, s.data(), &rcond, &rank, WORK.data(), &lwork, RWORK.data(), &info);
  lwork = int(WORK[0].real());
  WORK.resize(lwork);
  // Actually perform least squares.
  zgelss_(&nrow, &ncol, &nrhs, A, &nrow, B, &nrow, s.data(), &rcond, &rank, WORK.data(), &lwork, RWORK.data(), &info);
  return info;
}

// Using native fortran interface to lapack. Assumes arrays are in column major format.
inline int least_squares(double *A, double *B, int nrow, int ncol, int nrhs)
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
  return info;
}
inline int least_squares(std::complex<double> *A, std::complex<double> *B, int nrow, int ncol, int nrhs)
{
  char trans = 'N';
  int lwork = -1, info;
  std::vector<std::complex<double> > WORK(1);
  // Workspace query.
  zgels_(&trans, &nrow, &ncol, &nrhs, A, &nrow, B, &nrow, WORK.data(), &lwork, &info);
  lwork = int(WORK[0].real());
  WORK.resize(lwork);
  // Actually perform least squares.
  zgels_(&trans, &nrow, &ncol, &nrhs, A, &nrow, B, &nrow, WORK.data(), &lwork, &info);
  return info;
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

inline int cholesky(DistributedMatrix::Matrix<std::complex<double> > &A)
{
  char uplo = 'L';
  int info;
  pzpotrf_(&uplo, &A.nrows,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           &info);
  return info;
}

template <typename T>
inline void local_transpose(DistributedMatrix::Matrix<T> &A, bool row_major=true)
{
  std::vector<T> tmp(A.ncols*A.nrows);
  if (row_major) {
    for (int i = 0; i < A.nrows; i++) {
      for (int j = 0; j < A.ncols; j++) {
        tmp[i+j*A.nrows] = A.store[i*A.ncols+j];
      }
    }
  } else {
    for (int i = 0; i < A.nrows; i++) {
      for (int j = 0; j < A.ncols; j++) {
        tmp[i*A.ncols+j] = A.store[i+j*A.nrows];
      }
    }
  }
  A.store.swap(tmp);
  swap_dims(A);
}

template <typename T>
inline void swap_dims(DistributedMatrix::Matrix<T> &A)
{
  int tmp_row = A.nrows;
  A.nrows = A.ncols;
  A.ncols = tmp_row;
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

inline void transpose(DistributedMatrix::Matrix<std::complex<double> > &A, DistributedMatrix::Matrix<std::complex<double> > &AT, bool hermi=false)
{
  char trans;
  if (hermi) {
    trans = 'C';
  } else {
    trans = 'T';
  }
  std::complex<double> one = 1.0, zero = 0.0;
  pzgeadd_(&trans, &A.ncols, &A.nrows,
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

template <class T>
inline void initialise_descriptor(DistributedMatrix::Matrix<T> &A, ContextHandler::BlacsGrid &BG, int br, int bc)
{
  int irsrc = 0, icsrc = 0;
  if (BG.nprocs == 1) {
    Cblacs_gridinfo(BG.ctxt, &BG.nrows, &BG.ncols, &BG.row, &BG.col);
    if (BG.row == 0 && BG.col == 0) {
      A.local_nrows = A.nrows;
      A.local_ncols = A.ncols;
      descinit_(A.desc.data(), &A.nrows, &A.ncols, &A.nrows,
                &A.ncols, &irsrc, &icsrc, &BG.ctxt, &A.nrows,
                &A.info);
    } else {
      A.desc[1] = -1;
      A.local_nrows = 0;
      A.local_ncols = 0;
    }
  } else {
    A.local_nrows = numroc_(&A.nrows, &br, &BG.row, &A.izero, &BG.nrows);
    A.local_ncols = numroc_(&A.ncols, &bc, &BG.col, &A.izero, &BG.ncols);
    //if (BG.rank == 0) {
      //std::cout << "descinit: " << local_nrows << " " << local_nc << " " << nrows << " " << ncols << std::endl;
    //}
    A.lld = std::max(1, A.local_nrows);
    descinit_(A.desc.data(), &A.nrows, &A.ncols, &br,
              &bc, &irsrc, &icsrc, &BG.ctxt, &A.lld,
              &A.info);
  }
}

inline int svd(DistributedMatrix::Matrix<std::complex<double> > &A, std::vector<double> &S, ContextHandler::BlacsGrid &BG)
{
  char jobu = 'N', jobvt='N';
  int lwork = -1, info;
  std::vector<std::complex<double> > WORK(1);
  std::vector<double> RWORK(1);
  // Unused arrays for interface consistency.
  DistributedMatrix::Matrix<std::complex<double> > U(A.nrows, A.nrows, BG);
  DistributedMatrix::Matrix<std::complex<double> > VT(A.ncols, A.ncols, BG);
  // Workspace Query
  pzgesvd_(&jobu, &jobvt,
           &A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           S.data(),
           U.store.data(), &U.init_row_idx, &U.init_col_idx, U.desc.data(),
           VT.store.data(), &VT.init_row_idx, &VT.init_col_idx, VT.desc.data(),
           WORK.data(), &lwork, RWORK.data(),
           &info);
  // Actual computation
  lwork = int(WORK[0].real());
  WORK.resize(lwork);
  int lrwork = int(RWORK[0]);
  RWORK.resize(lrwork);
  pzgesvd_(&jobu, &jobvt,
           &A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           S.data(),
           U.store.data(), &U.init_row_idx, &U.init_col_idx, U.desc.data(),
           VT.store.data(), &VT.init_row_idx, &VT.init_col_idx, VT.desc.data(),
           WORK.data(), &lwork, RWORK.data(),
           &info);
  return info;
}

template <class T>
int rank(DistributedMatrix::Matrix<T> &A, ContextHandler::BlacsGrid &BG, bool write=false)
{
  // Singular Values.
  std::vector<double> S(A.nrows);
  svd(A, S, BG);
  // Criteria for singular value being numerically close to zero.
  // Same as is used in numpy/scipy and taken from Numerical Recipes.
  double rcond = std::max(A.nrows, A.ncols) * S[0] * 1e-16;
  // S is sorted so this is a bit stupid.
  // Binary search.
  if (BG.rank == 0 && write) {
    std::cout << "Singular values." << std::endl;
    for (int i = 0; i < S.size(); i++) {
      std::cout << i << " " << S[i] << std::endl;
    }
  }
  int null = 0;
  for (int i = 0; i < S.size(); i++) {
    if (S[i] < rcond) null++;
  }
  return A.nrows - null;
}

}
#endif
