#include <vector>
#include <iostream>
#include "H5Cpp.h"
#include <complex>

namespace H5Helper
{
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

  void write(H5::H5File &fh5, std::string name, std::vector<double> &data, std::vector<hsize_t> &dims)
  {
    H5::DataSpace dataspace(dims.size(), dims.data());
    H5::DataSet dataset = fh5.createDataSet(name.c_str(),
                                             H5::PredType::NATIVE_DOUBLE,
                                             dataspace);
    dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
  }

  void write(H5::H5File &fh5, std::string dset_name, std::vector<std::complex<double> > &data, std::vector<hsize_t> &dims)
  {
    // Create appropriate data type.
    H5::CompType complex_data_type(sizeof(data[0]));
    complex_data_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
    complex_data_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
    // Write data.
    H5::DataSpace dataspace(dims.size(), dims.data());
    H5::DataSet dataset = fh5.createDataSet(dset_name.c_str(),
                                             complex_data_type,
                                             dataspace);
    dataset.write(data.data(), complex_data_type);
  }
}
