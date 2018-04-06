#include <vector>
#include <iostream>
#include "H5Cpp.h"
#include <complex>

namespace H5Helper
{
  void read_dataset(H5::DataSet &dset, std::vector<double> &matrix, H5::DataSpace mspace, H5::DataSpace &dspace)
  {
    dset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE, mspace, dspace);
  }
  void read_dataset(H5::DataSet &dset, std::vector<int> &matrix, H5::DataSpace mspace, H5::DataSpace &dspace)
  {
    dset.read(matrix.data(), H5::PredType::NATIVE_INT, mspace, dspace);
  }
  void read_dataset(H5::DataSet &dset, std::vector<std::complex<double> > &matrix, H5::DataSpace mspace, H5::DataSpace &dspace)
  {
    H5::CompType complex_data_type(sizeof(matrix[0]));
    complex_data_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
    complex_data_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
    dset.read(matrix.data(), complex_data_type, mspace, dspace);
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
  void read_dims(std::string filename, std::string data_name, std::vector<hsize_t> &dims)
  {
    H5::H5File file = H5::H5File(filename.c_str(), H5F_ACC_RDONLY);
    const int ndims = 2;
    // get dataset
    H5::DataSet dataset = file.openDataSet(data_name.c_str());
    // get the dataspace
    H5::DataSpace dataspace = dataset.getSpace();
    // check that signal has 2 dims
    if (dataspace.getSimpleExtentNdims() != ndims) {
      std::cerr << "dataset has wrong number of dimensions" << std::endl;
    }
    // get dimensions
    dataspace.getSimpleExtentDims(dims.data(), NULL);
  }
}
