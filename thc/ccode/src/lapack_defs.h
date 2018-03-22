#ifndef LAPACK_DEFS_H
#define LAPACK_DEFS_H
extern "C" {
  void dgemm_(char *transa, char *transb, int *m, int *n, int *k, 
              double *alpha,
              double *A, int *lda,
              double *B, int *ldb,
              double *beta,
              double *C, int *ldc);
  // Parallel least squares solver.
  void dgels_(char *trans,
              int *m, int *n, int *nrhs,
              double *A, int *lda,
              double *B, int *ldb,
              double *WORK, int *lwork,
              int *info);
}
#endif
