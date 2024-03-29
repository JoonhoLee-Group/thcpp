#ifndef SCALAPACK_DEFS_H
#define SCALAPACK_DEFS_H
extern "C" {
  // Compute the number of rows and columns on processor locally.
  int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);
  // Initialise array descriptors.
  void descinit_(int *desc,
                 const int *m, const int *n, const int *mb, const int *nb,
                 const int *irsrc, const int *icsrc, const int *ictxt,
                 const int *lld, int *info);
  // Redistribute block cyclically distributed array from one blacs context to another.
  void pdgemr2d_(int *m, int *n,
                 double *a, int *ia, int *ja, int *desca,
                 double *b, int *ib, int *jb, int *descb,
                 int *ictxt);
  void pzgemr2d_(int *m, int *n,
                 std::complex<double> *a, int *ia, int *ja, int *desca,
                 std::complex<double> *b, int *ib, int *jb, int *descb,
                 int *ictxt);
  void pigemr2d_(int *m, int *n,
                 int *a, int *ia, int *ja, int *desca,
                 int *b, int *ib, int *jb, int *descb,
                 int *ictxt);
  // OK this is really (p)blas.
  // Matrix-Matrix multiplication.
  void pdgemm_(char * transa, char *transb, int *m, int *n, int *k,
               double *alpha,
               double *A, int* ia, int *ja, int *desca,
               double *B, int* ib, int *jb, int *descb,
               double *beta,
               double *C, int *ic, int *jc, int *descc);
  void pzgemm_(char * transa, char *transb, int *m, int *n, int *k,
               std::complex<double> *alpha,
               std::complex<double> *A, int* ia, int *ja, int *desca,
               std::complex<double> *B, int* ib, int *jb, int *descb,
               std::complex<double> *beta,
               std::complex<double> *C, int *ic, int *jc, int *descc);
  void pzherk_(char *uplo, char *transa, int *n, int *k,
               std::complex<double> *alpha,
               std::complex<double> *A, int* ia, int *ja, int *desca,
               std::complex<double> *beta,
               std::complex<double> *C, int *ic, int *jc, int *descc);
  // Parallel least squares solver.
  void pdgels_(char *trans,
               int *m, int *n, int *nrhs,
               double *A, int *ia, int *ja, int *desca,
               double *B, int *ib, int *jb, int *descb,
               double *WORK, int *lwork,
               int *info);
  void pzgels_(char *trans,
               int *m, int *n, int *nrhs,
               std::complex<double> *A, int *ia, int *ja, int *desca,
               std::complex<double> *B, int *ib, int *jb, int *descb,
               std::complex<double> *WORK, int *lwork,
               int *info);
  // Add two matrices.
  void pdgeadd_(char *trans, int *m, int *n,
                double *alpha,
                double *A, int *ia, int *ja, int *desca,
                double *beta,
                double *C, int *ic, int *jc, int *descc);
  // Add two matrices.
  void pzgeadd_(char *trans, int *m, int *n,
                std::complex<double> *alpha,
                std::complex<double> *A, int *ia, int *ja, int *desca,
                std::complex<double> *beta,
                std::complex<double> *C, int *ic, int *jc, int *descc);
  // Cholesky decomposition
  void pzpotrf_(char *uplo, int *n,
                std::complex<double> *A, int *ia, int *ja, int *desca,
                int *info);
  void pdgesvd_(char *jobu, char *jobvt,
                int *m, int *n,
                double *a, int *ia, int *ja, int *desca,
                double *s,
                double *u, int *iu, int *ju, int *descu,
                double *vt, int *ivt, int *jvt, int *descvt,
                double *work, int *lwork, double *rwork,
                int *info);
  void pzgesvd_(char *jobu, char *jobvt,
                int *m, int *n,
                std::complex<double> *a, int *ia, int *ja, int *desca,
                double *s,
                std::complex<double> *u, int *iu, int *ju, int *descu,
                std::complex<double> *vt, int *ivt, int *jvt, int *descvt,
                std::complex<double> *work, int *lwork, double *rwork,
                int *info);
  void pzgeqpf_(int *m, int *n,
                std::complex<double> *a, int *ia, int *ja, int *desca,
                int *ipiv,
                std::complex<double> *tau,
                std::complex<double> *work, int *lwork, double *rwork, int *lrwork,
                int *info);
  int indxl2g_(int *indxloc, int *nb, int *iproc, int *isrcproc, int *nprocs);
}
#endif
