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
  // Parallel least squares solver.
  void pdgels_(char *trans,
               int *m, int *n, int *nrhs,
               double *A, int *ia, int *ja, int *desca,
               double *B, int *ib, int *jb, int *descb,
               double *WORK, int *lwork,
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
}
#endif
