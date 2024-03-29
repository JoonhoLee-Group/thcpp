#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <assert.h>
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

template <typename T>
void add_constant(std::vector<T> &vec, T a)
{
  for (int i = 0; i < vec.size(); i++) {
    vec[i] += a;
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

inline bool is_zero(std::vector<double> &a, double thresh=1e-12)
{
  for (int i = 0; i < a.size(); i++) {
    if (std::abs(a[i]) > thresh)
      return false;
  }
  return true;
}

inline void product(int M, int N, int K,
                    double one,
                    double* A, int LDA,
                    double* B, int LDB,
                    double zero,
                    double* C, int LDC)
{
  char transa = 'N';
  char transb = 'N';

  // C = A*B -> fortran -> C' = B'*A',
  dgemm_(&transa, &transb, &M, &N, &K, &one, A, &LDA, B, &LDB, &zero, C, &LDC);
}

inline void product(DistributedMatrix::Matrix<std::complex<double> > &A,
                    DistributedMatrix::Matrix<std::complex<double> > &B,
                    DistributedMatrix::Matrix<std::complex<double> > &C,
                    char transA='N',
                    char transB='N')
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

inline void product(DistributedMatrix::Matrix<double> &A,
                    DistributedMatrix::Matrix<double> &B,
                    DistributedMatrix::Matrix<double> &C,
                    char transA='N',
                    char transB='N')
{
  double one = 1.0, zero = 0.0;
  int m, n, k;
  // m = # of rows of matrix op(A)
  // n = # of columns of matrix op(B)
  // k = # of columns/rows of matrix op(A)/op(B)
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
    n = B.nrows;
    k = A.nrows;
  } else {
    m = A.nrows;
    n = B.ncols;
    k = A.ncols;
  }
  pdgemm_(&transA, &transB, &m, &n, &k,
          &one,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
          B.store.data(), &B.init_row_idx, &B.init_col_idx, B.desc.data(),
          &zero,
          C.store.data(), &C.init_row_idx, &C.init_col_idx, C.desc.data());
}

// Rename this
inline void product(DistributedMatrix::Matrix<std::complex<double> > &A,
                    DistributedMatrix::Matrix<std::complex<double> > &C, char transA='N', char uplo='U')
{
  std::complex<double>  one = 1.0, zero = 0.0;
  pzherk_(&uplo, &transA, &A.ncols, &A.nrows,
          &one,
          A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
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
inline void swap_dims(DistributedMatrix::Matrix<T> &A)
{
  int tmp_row = A.nrows;
  A.nrows = A.ncols;
  A.ncols = tmp_row;
}


// Transpose matrix assumed to be on root processor (not distributed)
// row_major = true for data stored in C order : A[i][j] = A.store[i*A.ncols+j] 
// row_major = false for data stored in Fortran order : A[i][j] = A.store[i+j*A.nrows] 
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

inline void transpose(DistributedMatrix::Matrix<double> &A, DistributedMatrix::Matrix<double> &AT, bool hermi=false)
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

// TODO: Scoping issues.
template <typename T>
inline void transpose(DistributedMatrix::Matrix<T> &M, ContextHandler::BlacsGrid &Grid, bool hermi=false)
{
  // Temporary store for transposed matrix.
  DistributedMatrix::Matrix<T> MT(M.ncols, M.nrows, Grid);
  transpose(M, MT, hermi);
  M = MT;
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
inline void redistribute(int m, int n,
                         int *a, int ia, int ja, int *desca,
                         int *b, int ib, int jb, int *descb,
                         int ictxt)
{
    pigemr2d_(&m, &n, a, &ia, &ib, desca, b, &ia, &ib, descb, &ictxt);
}

template <typename T>
inline void redistribute(DistributedMatrix::Matrix<T> &M, ContextHandler::BlacsGrid &GridA, ContextHandler::BlacsGrid &GridB, bool verbose=false, int block_rows=-1, int block_cols=-1)
{
  if (GridB.rank == 0 && verbose) {
    std::cout << "  * Local shape (on root processor) before redistribution: (" << M.local_nrows << ", " << M.local_ncols << ")" << std::endl;
  }
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

template <typename T>
void down_sample_distributed_columns(DistributedMatrix::Matrix<T> &From, DistributedMatrix::Matrix<T> &To,
                                     std::vector<int> &indxs, ContextHandler::BlacsHandler &BH)
{
  // Figure out which columns to select on each processor.
  // We extend selected index array to be the same size as the CZt.ncols, so that we can
  // redistribute this array to align with CZt and don't need to figure out any indexing.
  DistributedMatrix::Matrix<int> ix_map(1, From.ncols, BH.Root);
  if (BH.rank == 0) {
    for (int i = 0; i < ix_map.store.size(); i++) {
        ix_map.store[i] = -1;
    }
    for (int i = 0; i < indxs.size(); i++) {
      ix_map.store[indxs[i]] = 1;
    }
  }
  // Redistribute to same processor grid as CZt.
  if (BH.rank == 0) {
    std::cout << " * Redistributing ix_map column cyclically." << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD); // Necessary?
  int ncols_per_block = ceil((double)From.ncols/BH.nprocs); // Check this, I think scalapack will take ceiling rather than floor.
  redistribute(ix_map, BH.Root, BH.Column, true, 1, ncols_per_block);
  int num_cols = 0;
  {
    for (int i = 0; i < ix_map.store.size(); i++) {
      if (ix_map.store[i] > 0) {
        // Work out number of selected columns on current processor.
        // These will not be evenly distributed amongst processors.
        num_cols++;
      }
    }
    std::vector<T> local_cols(From.nrows*num_cols);
    num_cols = 0;
    // Second time around copy data.
    for (int i = 0; i < ix_map.store.size(); i++) {
      if (ix_map.store[i] > 0) {
        // Index in original global array. We need this to sort collected To later.
        // From is stored in Fortran format, so columns are contiguous in memory, which is
        // what we want.
        std::copy(From.store.begin()+i*From.nrows,
                  From.store.begin()+(i+1)*From.nrows,
                  local_cols.begin()+num_cols*From.nrows);
        num_cols++;
      }
    }
    // Work out how many columns of data we'll receive from each processor.
    std::vector<int> recv_counts(BH.nprocs), disps(BH.nprocs);
    // Figure out number of columns each processor will send to root.
    MPI_Gather(&num_cols, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    disps[0] = 0;
    recv_counts[0] *= From.nrows;
    for (int i = 1; i < recv_counts.size(); i++) {
      disps[i] = disps[i-1] + recv_counts[i-1];
      recv_counts[i] *= From.nrows;
    }
    // Because interp_indxs is sorted and we've chunked From in an ordered way, then the
    // selected columns in local_cols will be places in To in such a way so as to match
    // the order in aoR_mu, the down sampled atomic orbitals at the interpolating points.
    // Not templated ....
    MPI_Gatherv(local_cols.data(), num_cols*To.nrows, MPI_DOUBLE_COMPLEX,
                To.store.data(), recv_counts.data(), disps.data(), MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
  } // Memory from local stores should be freed.
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
  int lwork = -1, info = 0;
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
  //zgesvd_(&jobu, &jobvt,
           //&A.nrows, &A.ncols,
           //A.store.data(), &A.nrows,
           //S.data(),
           //U.store.data(), &U.nrows,
           //VT.store.data(), &VT.nrows,
           //WORK.data(), &lwork, RWORK.data(),
           //&info);
  //std::cout << info << " " << WORK[0] << " " << RWORK[0] << " " << A.store.size() << " " << A.nrows << " " << A.ncols << std::endl;
  // Actual computation
  lwork = int(WORK[0].real());
  WORK.resize(lwork);
  int lrwork = int(RWORK[0]);
  RWORK.resize(lrwork);
  //RWORK.resize(5*std::min(A.nrows, A.ncols));
  //zgesvd_(&jobu, &jobvt,
           //&A.nrows, &A.ncols,
           //A.store.data(), &A.nrows,
           //S.data(),
           //U.store.data(), &U.nrows,
           //VT.store.data(), &VT.nrows,
           //WORK.data(), &lwork, RWORK.data(),
           //&info);
  pzgesvd_(&jobu, &jobvt,
           &A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           S.data(),
           U.store.data(), &U.init_row_idx, &U.init_col_idx, U.desc.data(),
           VT.store.data(), &VT.init_row_idx, &VT.init_col_idx, VT.desc.data(),
           WORK.data(), &lwork, RWORK.data(),
           &info);
  if (info < 0) {
    std::cout << " * WARNING: SVD Failed." << std::endl;
    std::cout << "  * " << -info << "th argument had illegal value." << std::endl;
    return -1;
  } else if (info > 0) {
    std::cout << " * WARNING: SVD Failed." << std::endl;
    std::cout << "  * " << "ZBDSQR did not converge." << std::endl;
    if (info == std::min(A.nrows, A.ncols)+1) {
      std::cout << "  * Eigenvalues are not consistent across processor grid." << std::endl;
      std::cout << "  * Result cannot be trusted." << std::endl;
    }
    return -1;
  } else {
    return info;
  }
}

inline int svd(DistributedMatrix::Matrix<double> &A,
               DistributedMatrix::Matrix<double>& U,
               DistributedMatrix::Matrix<double>& VT,
               std::vector<double> &S,
               ContextHandler::BlacsGrid &BG)
{
  char jobu = 'V', jobvt='V';
  int lwork = -1, info = 0;
  std::vector<double> WORK(1);
  std::vector<double> RWORK(1);
  // Unused arrays for interface consistency.
  //DistributedMatrix::Matrix<double> U(A.nrows, A.nrows, BG);
  //DistributedMatrix::Matrix<double> VT(A.ncols, A.ncols, BG);
  // Workspace Query
  pdgesvd_(&jobu, &jobvt,
           &A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           S.data(),
           U.store.data(), &U.init_row_idx, &U.init_col_idx, U.desc.data(),
           VT.store.data(), &VT.init_row_idx, &VT.init_col_idx, VT.desc.data(),
           WORK.data(), &lwork, RWORK.data(),
           &info);
  if (info != 0) {
    std::cout << " * Error determining optimal workspace." << std::endl;
    std::cout << " * Error : " << info << std::endl;
  }
  // Actual computation
  lwork = int(WORK[0]);
  WORK.resize(lwork);
  int lrwork = int(RWORK[0]);
  RWORK.resize(lrwork);
  pdgesvd_(&jobu, &jobvt,
           &A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           S.data(),
           U.store.data(), &U.init_row_idx, &U.init_col_idx, U.desc.data(),
           VT.store.data(), &VT.init_row_idx, &VT.init_col_idx, VT.desc.data(),
           WORK.data(), &lwork, RWORK.data(),
           &info);
  if (info < 0) {
    std::cout << " * WARNING: SVD Failed." << std::endl;
    std::cout << "  * " << -info << "th argument had illegal value." << std::endl;
    return -1;
  } else if (info > 0) {
    std::cout << " * WARNING: SVD Failed." << std::endl;
    std::cout << "  * " << "DBDSQR did not converge." << std::endl;
    if (info == std::min(A.nrows, A.ncols)+1) {
      std::cout << "  * Eigenvalues are not consistent across processor grid." << std::endl;
      std::cout << "  * Result cannot be trusted." << std::endl;
    }
    return -1;
  } else {
    return info;
  }
}

inline int pseudo_inverse(DistributedMatrix::Matrix<double> &A,
                          DistributedMatrix::Matrix<double> &Ainv,
                          double rcond,
                          ContextHandler::BlacsHandler &BH)
{
  int nrows = A.nrows;
  int ncols = A.ncols;
  DistributedMatrix::Matrix<double> U(nrows,ncols,BH.Square,A.block_nrows,A.block_ncols);
  DistributedMatrix::Matrix<double> VT(nrows,ncols,BH.Square,A.block_nrows,A.block_ncols);
  std::vector<double> S(std::min(nrows, ncols), 0.0);
  // 1. Perform SVD.
  svd(A, U, VT, S, BH.Square);
  // 2. Discard small singular values.
  double thresh = rcond * S[0];
  int rank = 0;
  for (int i = 0; i < S.size(); i++) {
    if (std::abs(S[i]) > thresh) {
      S[i] = 1.0 / S[i];
      rank++;
    } else {
      S[i] = 0.0;
    }
  }
  //DistributedMatrix::Matrix<double> UU("numpy_svd.h5", "u", BH.Root, true, false);
  //DistributedMatrix::Matrix<double> VVT("numpy_svd.h5", "vt", BH.Root, true, false);
  //DistributedMatrix::Matrix<double> SMat("numpy_svd.h5", "s", BH.Root, true, false);
  //DistributedMatrix::Matrix<double> Pinv("numpy_svd.h5", "sinv", BH.Root, true, false);
  //if (BH.rank == 0) 
  //{
    //MatrixOperations::local_transpose(UU);
    //MatrixOperations::swap_dims(UU);
    //MatrixOperations::local_transpose(VVT);
    //MatrixOperations::swap_dims(VVT);
    //MatrixOperations::local_transpose(Pinv);
    //MatrixOperations::swap_dims(Pinv);
  //}
  //MatrixOperations::redistribute(UU, BH.Root, BH.Square, true);
  //MatrixOperations::redistribute(VVT, BH.Root, BH.Square, true);
  //MatrixOperations::redistribute(SMat, BH.Root, BH.Square, true);
  // 3. Form pinv
  DistributedMatrix::Matrix<double> SMat(nrows,ncols,BH.Root,A.block_nrows,A.block_ncols);
  if (BH.rank == 0) {
    for (int i = 0; i < S.size(); i++)
      SMat.store[i+S.size()*i] = S[i];
  }
  redistribute(SMat, BH.Root, BH.Square);
  DistributedMatrix::Matrix<double> T1(nrows,ncols,BH.Square,A.block_nrows,A.block_ncols);
  // Ainv = (VT)^{T} S^{-1} U^T
  product(SMat, U, T1, 'N', 'T');
  product(VT, T1, Ainv, 'T', 'N');
  //MatrixOperations::redistribute(Ainv, BH.Square, BH.Root);
  //if (BH.rank == 0) {
    //std::cout << Ainv.store[7*Ainv.nrows]-Pinv.store[7*Pinv.nrows] << std::endl;
    //for (int i = 0; i < Ainv.store.size(); i++)
      //if (std::abs(Ainv.store[i] - Pinv.store[i])>1e-10) {
        //std::cout << i << " " << Ainv.store[i]-Pinv.store[i] << std::endl;
      //}
  //}
  //Ainv.store = Pinv.store;
  //MatrixOperations::redistribute(Ainv, BH.Root, BH.Square);
  return rank;
}

// T[L,mn] = P[L,m] P[L,n]
// distributed over L
inline void tensor_rank_one(DistributedMatrix::Matrix<double> &P,
                           DistributedMatrix::Matrix<double> &T)
{
  int mn = P.nrows * P.nrows;
  int m = P.nrows;
  double alpha = 1.0;
  int inc = 1;
  for (int l = 0; l < T.local_ncols; l++) {
    dger_(&m, &m, &alpha,
          P.store.data()+l*m, &inc,
          P.store.data()+l*m, &inc,
          T.store.data()+l*mn, &m);
    //if (BH.rank == 0) {
      //for (int i = 0; i < mn; i++)
        //std::cout << *(T.store.data()+l*mn+i) << std::endl;
      //std::cout << std::endl;
    //}
  }
  MPI_Barrier(MPI_COMM_WORLD);
  //std::cout << "done: " << std::endl;
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
      std::cout << "SV " << i << " " << S[i] << std::endl;
    }
  }
  int null = 0;
  for (int i = 0; i < S.size(); i++) {
    if (S[i] < rcond) null++;
  }
  return A.nrows - null;
}

template<typename T>
void hadamard_product(DistributedMatrix::Matrix<T> &A)
{
  for (int i = 0; i < A.store.size(); i++)
    A.store[i] = A.store[i] * A.store[i];
}

template<typename T>
void hadamard_product(DistributedMatrix::Matrix<T> &A,
                      DistributedMatrix::Matrix<T>& B,
                      DistributedMatrix::Matrix<T>& C)
{
  assert(A.store.size() == B.store.size());
  assert(C.store.size() == B.store.size());
  for (int i = 0; i < A.store.size(); i++)
    C.store[i] = A.store[i] * B.store[i];
}

// QR decomposition with column pivoting.
// Currently only returns permutation vector.
template <class T>
int qrcp(DistributedMatrix::Matrix<T> &A, std::vector<int> &perm,
         ContextHandler::BlacsGrid &BG, bool write=false)
{
  perm.resize(A.ncols);
  std::vector<std::complex<double> > TAU(std::min(A.nrows, A.ncols));
  std::vector<std::complex<double> > WORK(1);
  std::vector<double> RWORK(1);
  int lwork = -1, lrwork = -1;
  int info;
  // First perform workspace query.
  pzgeqpf_(&A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           perm.data(),
           TAU.data(),
           WORK.data(), &lwork, RWORK.data(), &lrwork,
           &info);
  lwork = int(WORK[0].real());
  WORK.resize(lwork);
  lrwork = int(RWORK[0]);
  RWORK.resize(lrwork);
  // Perform QRCP decomposition.
  pzgeqpf_(&A.nrows, &A.ncols,
           A.store.data(), &A.init_row_idx, &A.init_col_idx, A.desc.data(),
           perm.data(),
           TAU.data(),
           WORK.data(), &lwork, RWORK.data(), &lrwork,
           &info);
  // perm will be fortran indexed so subtract one.
  add_constant(perm, -1);
  return info;
}

inline int global_matrix_col_index(int local_index, ContextHandler::BlacsGrid &BG, int nb)
{
  int root = 0;
  int li = local_index + 1; // C to fortran
  return indxl2g_(&li, &nb, &BG.col, &root, &BG.ncols) - 1; // fortran to C
}

inline int global_matrix_row_index(int local_index, ContextHandler::BlacsGrid &BG, int mb)
{
  int root = 0;
  int li = local_index + 1; // C to fortran
  return indxl2g_(&li, &mb, &BG.row, &root, &BG.nrows) - 1; // fortran to C
}

template <typename T>
void print_matrix(DistributedMatrix::Matrix<T>& M, bool row_major)
{
  if (row_major) {
    for (int i = 0; i < M.nrows; i++) {
      for (int j = 0; j < M.ncols; j++) {
        std::cout << M.store[i*M.ncols+j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  } else {
    for (int j = 0; j < M.ncols; j++) {
      for (int i = 0; i < M.nrows; i++) {
        std::cout << M.store[j*M.nrows+i] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

}
#endif
