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
  void zgels_(char *trans,
              int *m, int *n, int *nrhs,
              std::complex<double> *A, int *lda,
              std::complex<double> *B, int *ldb,
              std::complex<double> *WORK, int *lwork,
              int *info);
  void zgelss_(int *nrow, int *ncol, int *nrhs,
               std::complex<double> *A, int *lda, std::complex<double> *B, int *ldb,
               double *S, double *rcond, int *rank,
               std::complex<double> *WORK, int *lwork, double *RWORK,
               int *info);
  void zgeqpf_(int *m, int *n,
               std::complex<double> *a, int *lda,
               int *ipiv,
               std::complex<double> *tau,
               std::complex<double> *work, double *rwork,
               int *info);
  void zgesvd_(char *jobu, char *jobvt,
               int *m, int *n, std::complex<double> *a, int *lda,
               double *s,
               std::complex<double> *u, int *ldu,
               std::complex<double> *vt, int *ldvt,
               std::complex<double> *work, int *lwork, double *rwork,
               int *info);
}
#endif
