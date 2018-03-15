#include <vector>
#include <iostream>
#include "H5Cpp.h"

namespace H5Helper 
{
const int H5ERROR = 11;

void read_matrix(H5::H5File matrix_file, const H5std_string data_name,
                  std::vector<double> &matrix, int &nrow, int &ncol)
{
  const int ndims = 2;
  hsize_t dims[2];

  // get signal dataset
  H5::DataSet dataset = matrix_file.openDataSet(data_name);

  // check that signal is float
  if (dataset.getTypeClass() != H5T_FLOAT) {
    std::cerr << "signal dataset has wrong type" << std::endl;
  }

  // check that signal is double
  if (dataset.getFloatType().getSize() != sizeof(double)) {
    std::cerr << "signal dataset has wrong type size" << std::endl;
  }

  // get the dataspace
  H5::DataSpace dataspace = dataset.getSpace();
  // check that signal has 2 dims
  if (dataspace.getSimpleExtentNdims() != ndims) {
    std::cerr << "signal dataset has wrong number of dimensions" << std::endl;
  }

  // get dimensions
  dataspace.getSimpleExtentDims(dims, NULL);

  // allocate memory and read data
  nrow = dims[0];
  ncol = dims[1];
  matrix.resize(dims[0]*dims[1]);

  H5::DataSpace data_mspace(ndims, dims);
  dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE, data_mspace, dataspace);
}

void read_matrices(std::vector<double> &CZt, std::vector<double> &CCt,
                   int &nmu, int &ngrid)
{
  H5std_string filename = "thc_data.h5";
  // open file
  H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);
  read_matrix(file, "CZt", CZt, nmu, ngrid);
  read_matrix(file, "CCt", CCt, nmu, nmu);
  // all done with file
  file.close();
}

void write_interpolating_points(std::vector<double> &IPTS, int nmu, int ngrid)
{
  H5std_string filename = "thc_interpolating_vectors.h5";
  H5std_string dataset_name = "interpolating_vectors";
  // open file
  H5::H5File file = H5::H5File(filename, H5F_ACC_TRUNC);
  hsize_t dims[2];
  dims[0] = nmu;
  dims[1] = ngrid;
  H5::DataSpace dataspace(2, dims);
  H5::DataSet dataset = file.createDataSet(dataset_name,
                                           H5::PredType::NATIVE_DOUBLE,
                                           dataspace);
  dataset.write(IPTS.data(), H5::PredType::NATIVE_DOUBLE); 
}
}
