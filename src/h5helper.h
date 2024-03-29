#ifndef READ_MATRIX_H
#define READ_MATRIX_H
#include "H5Cpp.h"
#include <vector>
#include <complex>
namespace H5Helper
{
  void read_dataset(H5::DataSet &dset, std::vector<double> &matrix, H5::DataSpace mspace, H5::DataSpace &dspace);
  void read_dataset(H5::DataSet &dset, std::vector<std::complex<double> > &matrix, H5::DataSpace mspace, H5::DataSpace &dspace);
  void read_dims(std::string filename, std::string data_name, std::vector<hsize_t> &dims);
  void read_dataset(H5::DataSet &dset, std::vector<int> &matrix, H5::DataSpace mspace, H5::DataSpace &dspace);

  template <typename T>
  void read_matrix(H5::H5File matrix_file, const H5std_string data_name,
                    std::vector<T> &matrix, std::vector<hsize_t> &dims)
  {
    const int ndims = 2;

    // get dataset
    H5::DataSet dataset = matrix_file.openDataSet(data_name);
    // get the dataspace
    H5::DataSpace dataspace = dataset.getSpace();
    // check that signal has 2 dims
    if (dataspace.getSimpleExtentNdims() != ndims) {
      std::cerr << "dataset has wrong number of dimensions" << std::endl;
    }

    // get dimensions
    dataspace.getSimpleExtentDims(dims.data(), NULL);

    // allocate memory and read data
    matrix.resize(dims[0]*dims[1]);

    H5::DataSpace data_mspace(ndims, dims.data());
    //dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE, data_mspace, dataspace);
    read_dataset(dataset, matrix, data_mspace, dataspace);
  }
  template <typename T>
  void read_matrix_hyperslab(H5::H5File matrix_file, const H5std_string data_name,
                             std::vector<T> &matrix, std::vector<hsize_t> &dims,
                             int iproc, int nprocs, hsize_t &chunk)
  {
    const int ndims = 2;

    // get dataset
    H5::DataSet dataset = matrix_file.openDataSet(data_name);
    // get the dataspace
    H5::DataSpace dataspace = dataset.getSpace();
    // check that signal has 2 dims
    if (dataspace.getSimpleExtentNdims() != ndims) {
      std::cerr << "dataset has wrong number of dimensions" << std::endl;
    }

    // get dimensions
    dataspace.getSimpleExtentDims(dims.data(), NULL);
    hsize_t nrows = dims[0], ncols = dims[1];
    chunk = ceil((double)nrows/nprocs);
    hsize_t row_offset = iproc * chunk;
    hsize_t nrows_iproc = std::min(nrows-row_offset, chunk);
    //std::cout << nrows << " " << ncols << " " << chunk << " " << (double)(nrows/nprocs) << " " << row_offset << " " << nrows_iproc << " " << nprocs << std::endl;
    // allocate memory and read data
    matrix.resize(nrows_iproc*dims[1]);
    std::vector<hsize_t> count = {nrows_iproc, dims[1]};
    std::vector<hsize_t> offset = {row_offset, 0};
    dataspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());
    std::vector<hsize_t> dims_out = {nrows_iproc, dims[1]};
    std::vector<hsize_t> offset_out = {0, 0};
    std::vector<hsize_t> count_out = {nrows_iproc, dims[1]};
    H5::DataSpace memspace(ndims, dims_out.data());
    memspace.selectHyperslab(H5S_SELECT_SET, count_out.data(), offset_out.data());
    read_dataset(dataset, matrix, memspace, dataspace);
  }
  void write(H5::H5File &fh5, std::string name, std::vector<std::complex<double> > &data, std::vector<hsize_t> &dims, bool compress=false);
  void write(H5::H5File &fh5, std::string name, std::vector<double> &data, std::vector<hsize_t> &dims, bool compress=false);
  void write(H5::H5File &fh5, std::string name, std::vector<int> &data, std::vector<hsize_t> &dims, bool compress=false);
  void create_gzip_plist(H5::DSetCreatPropList &plist, std::vector<hsize_t> &dims, hsize_t chunk_size);
}
#endif
