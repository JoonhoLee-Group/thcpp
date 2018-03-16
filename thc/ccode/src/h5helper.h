#ifndef READ_MATRIX_H
#define READ_MATRIX_H
#include "H5Cpp.h"
#include <vector>
namespace H5Helper
{
void read_matrix(H5::H5File matrix_file, const H5std_string data_name,
                  std::vector<double> &matrix, std::vector<hsize_t> &dims);
void read_matrices(std::vector<double> &CZt, std::vector<double> &CCt,
                   int &nmu, int &ngrid, bool &row_major);
void write_interpolating_points(std::vector<double> &IPTS, int nmu, int ngrid);
}
#endif
