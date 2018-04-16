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
    // h5py format for complex numbers
    H5::CompType complex_data_type(sizeof(matrix[0]));
    complex_data_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
    complex_data_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
    dset.read(matrix.data(), complex_data_type, mspace, dspace);
  }

  void create_gzip_plist(H5::DSetCreatPropList &plist, std::vector<hsize_t> &dims, hsize_t chunk_size) {
    std::vector<hsize_t> cdims(dims.size());
    for (int i = 0; i < dims.size(); i++) {
      cdims[i] = std::min(dims[i], chunk_size);
    }
    plist.setChunk(cdims.size(), cdims.data());
    plist.setDeflate(6);
  }

  void write(H5::H5File &fh5, std::string name, std::vector<double> &data, std::vector<hsize_t> &dims, bool compress)
  {
    H5::DataSpace dataspace(dims.size(), dims.data());
    H5::DSetCreatPropList plist;
    hsize_t chunk_size = 20;
    if (compress) create_gzip_plist(plist, dims, chunk_size);
    H5::DataSet dataset = fh5.createDataSet(name.c_str(),
                                            H5::PredType::NATIVE_DOUBLE,
                                            dataspace,
                                            plist);
    dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
  }

  void write(H5::H5File &fh5, std::string name, std::vector<int> &data, std::vector<hsize_t> &dims, bool compress)
  {
    H5::DataSpace dataspace(dims.size(), dims.data());
    H5::DSetCreatPropList plist;
    hsize_t chunk_size = 20;
    if (compress) create_gzip_plist(plist, dims, chunk_size);
    H5::DataSet dataset = fh5.createDataSet(name.c_str(),
                                            H5::PredType::NATIVE_INT,
                                            dataspace,
                                            plist);
    dataset.write(data.data(), H5::PredType::NATIVE_INT);
  }

  void write(H5::H5File &fh5, std::string name, std::vector<std::complex<double> > &data, std::vector<hsize_t> &dims, bool compress)
  {
    // QMCPACK complex number format.
    dims.push_back(2);
    H5::DataSpace dataspace(dims.size(), dims.data());
    H5::DSetCreatPropList plist;
    hsize_t chunk_size = 20;
    if (compress) create_gzip_plist(plist, dims, chunk_size);
    H5::DataSet dataset = fh5.createDataSet(name.c_str(),
                                            H5::PredType::NATIVE_DOUBLE,
                                            dataspace,
                                            plist);
    dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
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
