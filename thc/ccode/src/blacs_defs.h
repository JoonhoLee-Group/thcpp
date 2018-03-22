#ifndef BLACS_DEFS_H
#define BLACS_DEFS_H
extern "C" {
  void Cblacs_pinfo(int* myid, int* numproc);
  void Cblacs_get(int, int, int* ctxt);
  void Cblacs_gridinit(int* ctxt, const char* order, int nprow, int npcol);
  void Cblacs_gridinfo(int ctxt, int *nprow, int *npcol, int *myrow, int *mycol);
  void Cblacs_pcoord(int ctxt, int myid, int* myrow, int* mycol);
  void Cblacs_gridexit(int ctxt);
}
#endif
