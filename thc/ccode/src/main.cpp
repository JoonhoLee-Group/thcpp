#include <iostream>
#include <vector>
#include <iomanip>
#include <time.h>
#include "h5helper.h"
#include "utils.h"
#include "matrix_operations.h"

int main(int argc, char* argv[])
{
  std::vector<double> CZt, CCt, X;
  int nmu, ngrid;
  double sum1 = 0, sum2 = 0;
  const double one = 1.0, zero = 0.0;
  bool row_major;

  std::cout << "Reading CZt and CCt matrices." << std::endl;
  double tread = clock();
  H5Helper::read_matrices(CZt, CCt, nmu, ngrid, row_major);
  tread = clock() - tread;
  std::cout << "Time taken to read matrices: " << " " << tread / CLOCKS_PER_SEC << " seconds" << std::endl;
  double mem_CCt = UTILS::get_memory(CCt); 
  double mem_CZt = UTILS::get_memory(CZt); 
  std::cout << "Memory usage for CCt: " << mem_CCt << " GB" << std::endl;
  std::cout << "Memory usage for CZt: " << mem_CZt << " GB" << std::endl;
  std::cout << "Total memory usage: " << mem_CCt + mem_CZt << " GB" << std::endl;
  double tlsq = clock();
  std::cout << "Performing serial least squares solve." << std::endl;
  MatrixOperations::least_squares(CCt.data(), CZt.data(), nmu, nmu, ngrid);
  tlsq = clock() - tlsq;
  std::cout << "Time for serial least squares solve : " << tlsq / CLOCKS_PER_SEC << " seconds" << std::endl;
  H5Helper::write_interpolating_points(CZt, nmu, ngrid);
}
