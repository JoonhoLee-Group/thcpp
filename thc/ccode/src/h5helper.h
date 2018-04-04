#ifndef READ_MATRIX_H
#define READ_MATRIX_H
#include "H5Cpp.h"
#include <vector>
#include <complex>
namespace H5Helper
{
  void read_matrix(H5::H5File matrix_file, const H5std_string data_name, std::vector<double> &matrix, std::vector<hsize_t> &dims);
  void write(H5::H5File &fh5, std::string name, std::vector<std::complex<double> > &data, std::vector<hsize_t> &dims);
  void write(H5::H5File &fh5, std::string name, std::vector<double> &data, std::vector<hsize_t> &dims);
}
#endif
