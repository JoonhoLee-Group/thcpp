#include <vector>
#include <iostream>
#include "H5Cpp.h"
#include <complex>

namespace H5Helper
{
const int H5ERROR = 11;

void read_matrix(H5::H5File matrix_file, const H5std_string data_name,
                  std::vector<double> &matrix, std::vector<hsize_t> &dims)
{
  const int ndims = 2;

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
  dataspace.getSimpleExtentDims(dims.data(), NULL);

  // allocate memory and read data
  matrix.resize(dims[0]*dims[1]);

  H5::DataSpace data_mspace(ndims, dims.data());
  dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE, data_mspace, dataspace);
}

void read_matrices(std::vector<double> &CZt, std::vector<double> &CCt,
                   int &nmu, int &ngrid, bool &row_major)
{
  std::vector<hsize_t> dims_CZt(2), dims_CCt(2);
  H5std_string filename = "thc_data.h5";
  H5::H5File file = H5::H5File(filename, H5F_ACC_RDONLY);
  read_matrix(file, "CZt", CZt, dims_CZt);
  read_matrix(file, "CCt", CCt, dims_CCt);
  if (dims_CZt[0] > dims_CZt[1]) {
    std::cout << "Matrices are in FORTRAN / column major format." << std::endl;
    nmu = dims_CCt[0];
    ngrid = dims_CZt[0];
    row_major = false;
  } else {
    std::cout << "Matrices are in C / row major format." << std::endl;
    nmu = dims_CCt[0];
    ngrid = dims_CZt[1];
    row_major = true;
  }
  file.close();
}

void write_interpolating_points(std::vector<double> &IPTS, int nmu, int ngrid)
{
  H5std_string filename = "thc_interpolating_vectors.h5";
  H5std_string dataset_name = "interpolating_vectors";
  // open file
  H5::H5File file = H5::H5File(filename, H5F_ACC_TRUNC);
  hsize_t dims[2];
  dims[1] = nmu;
  dims[0] = ngrid;
  H5::DataSpace dataspace(2, dims);
  H5::DataSet dataset = file.createDataSet(dataset_name,
                                           H5::PredType::NATIVE_DOUBLE,
                                           dataspace);
  dataset.write(IPTS.data(), H5::PredType::NATIVE_DOUBLE);
}

void write_fft(std::vector<std::complex<double> > &fft, int nmu, int ngrid)
{
  H5std_string filename = "fft_data.h5";
  H5std_string dataset_name = "fftd_points";
  // open file
  H5::CompType complex_data_type(sizeof(fft[0]));
  complex_data_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
  complex_data_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
  H5::H5File file = H5::H5File(filename, H5F_ACC_TRUNC);
  hsize_t dims[2];
  dims[0] = nmu;
  dims[1] = ngrid;
  H5::DataSpace dataspace(2, dims);
  H5::DataSet dataset = file.createDataSet(dataset_name,
                                           complex_data_type,
                                           dataspace);
  dataset.write(fft.data(), complex_data_type);
}
}
