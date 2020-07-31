#include <iostream>

#include "distributed_matrix.h"
#include "context_handler.h"
#include "matrix_operations.h"
#include "molecular.h"
#include "logging.h"

void generate_qmcpack_molecular_thc(std::string input_file,
                                    std::string output_file,
                                    bool need_density_fitting,
                                    bool verbose)
{
  // 1. Parse data from hdf5
  ContextHandler::BlacsHandler BH;
  // Assuming all data from hdf5 is C ordered.
  bool row_major_read = true;
  bool parallel_read = false;
  DistributedMatrix::Matrix<double> orbs(input_file, "etaPm", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack which expects fortran order
    MatrixOperations::local_transpose(orbs);
    MatrixOperations::swap_dims(orbs);
  }
  MatrixOperations::redistribute(orbs, BH.Root, BH.Square, verbose);
  int nthc = orbs.nrows;
  int norb = orbs.ncols;
  DistributedMatrix::Matrix<double> Sinv(nthc, nthc, BH.Square);
  find_pseudo_inverse(orbs, BH, Sinv);
  //MatrixOperations::redistribute(Sinv, BH.Square, BH.Root);
  //Logging::write_matrix(Sinv, "pinv.h5", "pinv", "", BH.rank==0, true, false, true); 
  //MatrixOperations::redistribute(Sinv, BH.Root, BH.Square);
  // Get intermediate BxL
  DistributedMatrix::Matrix<double> BxL;
  if (need_density_fitting) {
    construct_BxL(input_file, orbs, BH, BxL);
  } else {
    get_BxL(input_file, BH, BxL);
  }
  // Construct DxK
  int naux = BxL.ncols;
  DistributedMatrix::Matrix<double> DxK(naux, nthc, BH.Square);
  MatrixOperations::product(BxL, Sinv, DxK);
  DistributedMatrix::Matrix<double> Muv(nthc, nthc, BH.Square);
  MatrixOperations::product(DxK, DxK, Muv, 'T', 'N');
  // Write_qmcpack_data
  // 1. Rotate to MO basis.
  DistributedMatrix::Matrix<double> hcore(input_file, "hcore", BH.Root, row_major_read, parallel_read);
  DistributedMatrix::Matrix<double> mo_coeff(input_file, "mo_coeff", BH.Root, row_major_read, parallel_read);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack which expects fortran order
    MatrixOperations::local_transpose(hcore);
    MatrixOperations::swap_dims(hcore);
    MatrixOperations::local_transpose(mo_coeff);
    MatrixOperations::swap_dims(mo_coeff);
  }
  Logging::log("Redistributing hcore", BH.rank==0);
  MatrixOperations::redistribute(hcore, BH.Root, BH.Square, verbose);
  Logging::log("Redistributing mo_coeff", BH.rank==0);
  MatrixOperations::redistribute(mo_coeff, BH.Root, BH.Square, verbose);
  DistributedMatrix::Matrix<double> T1(hcore.nrows, hcore.ncols, BH.Square);
  // 1.a rotate hcore to mo basis
  Logging::log("Tranforming to MO basis", BH.rank==0);
  MatrixOperations::product(hcore, mo_coeff, T1);
  MatrixOperations::product(mo_coeff, T1, hcore, 'T', 'N');
  // 1.b rotate orbitals to mo basis
  {
    DistributedMatrix::Matrix<double> orbs_(orbs.nrows, orbs.ncols, BH.Square);
    MatrixOperations::product(orbs, mo_coeff, orbs_);
    std::copy_n(orbs_.store.data(), orbs_.store.size(), orbs.store.data());
  }
  std::vector<int> nelec(2);
  std::vector<double> enuc(2);
  if (BH.rank == 0) {
    std::vector<hsize_t> dims(2);
    // Overwrite any existing file
    H5::H5File out = H5::H5File(output_file.c_str(), H5F_ACC_TRUNC);
    H5::H5File fin = H5::H5File(input_file.c_str(), H5F_ACC_RDWR);
    dims[0] = 2;
    Logging::log("Reading nelec", BH.rank==0);
    H5Helper::read_matrix(fin, "nelec", nelec, dims);
    Logging::log("Reading enuc", BH.rank==0);
    H5Helper::read_matrix(fin, "enuc", enuc, dims);
  }
  DistributedMatrix::Matrix<double> orbs_half(orbs);
  Logging::log("Writing QMCPACK data", BH.rank==0);
  write_qmcpack_hdf5(output_file, hcore, nelec, enuc, orbs, orbs_half, DxK, Muv, BH);
}

void write_qmcpack_hdf5(std::string output_file,
                        DistributedMatrix::Matrix<double>& hcore,
                        std::vector<int> nelec,
                        std::vector<double> enuc,
                        DistributedMatrix::Matrix<double>& orbs,
                        DistributedMatrix::Matrix<double>& orbs_half,
                        DistributedMatrix::Matrix<double>& luv,
                        DistributedMatrix::Matrix<double>& muv_half,
                        ContextHandler::BlacsHandler BH)
{
  // THC DATA
  MatrixOperations::redistribute(orbs_half, BH.Square, BH.Root, true);
  int nmo = orbs_half.ncols;
  int nthc = orbs_half.nrows;
  DistributedMatrix::Matrix<double> orbs_occ(nelec[0], nthc, BH.Root);
  // orbs is fortran ordered [Nmu,M] so first Nmu*N (occupieds) are contiguous
  if (BH.rank == 0) {
    std::copy_n(orbs_half.store.data(), nelec[0]*nthc, orbs_occ.store.data());
    std::cout << orbs_occ.store[0] << std::endl;
  }
  // Need to write [M,Nmu], but orbs is already transposed from C/HDF5 point of view, so
  // just swap dims
  Logging::write_matrix(orbs_half, output_file, "HalfTransformedFullOrbitals", "THC", BH.rank==0, false, false, true);
  // These are already stored correctly
  Logging::write_matrix(orbs_occ, output_file, "HalfTransformedOccOrbitals", "THC", BH.rank==0);
  MatrixOperations::redistribute(muv_half, BH.Square, BH.Root, true);
  Logging::write_matrix(muv_half, output_file, "HalfTransformedMuv", "THC", BH.rank==0, true);
  MatrixOperations::redistribute(orbs, BH.Square, BH.Root, true);
  Logging::write_matrix(orbs, output_file, "Orbitals", "THC", BH.rank==0, false, false, true);
  MatrixOperations::redistribute(luv, BH.Square, BH.Root, true);
  Logging::write_matrix(luv, output_file, "Luv", "THC", BH.rank==0, false, false, true);
  // Regular data
  MatrixOperations::redistribute(hcore, BH.Square, BH.Root, true);
  Logging::write_matrix(hcore, output_file, "hcore", "", BH.rank==0, true);
  if (BH.rank == 0) {
    H5::Exception::dontPrint();
    H5::H5File file = H5::H5File(output_file.c_str(), H5F_ACC_RDWR);
    H5::Group base = file.openGroup("/Hamiltonian");
    int nmo = hcore.nrows;
    int nthc = luv.nrows;
    int nthc_half = muv_half.nrows;
    std::vector<hsize_t> dims(1);
    dims[0] = 3;
    std::vector<int> thc_dims = {nmo, nthc, nthc_half};
    H5Helper::write(file, "/Hamiltonian/THC/dims", thc_dims, dims);
    std::vector<int> qmcpack_dims = {0, 0, 0, nmo, nelec[0], nelec[1], 0, 0};
    dims[0] = qmcpack_dims.size();
    H5Helper::write(file, "/Hamiltonian/dims", qmcpack_dims, dims);
    dims[0] = 2;
    H5Helper::write(file, "/Hamiltonian/Energies", enuc, dims);
  }
}

void construct_BxL(std::string input_file,
                   DistributedMatrix::Matrix<double>& orbs,
                   ContextHandler::BlacsHandler& BH,
                   DistributedMatrix::Matrix<double>& BxL)
{
  bool parallel_read = false;
  DistributedMatrix::Matrix<double> df_ints(input_file, "eri_df", BH.Root, true, false);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack
    MatrixOperations::local_transpose(df_ints);
    MatrixOperations::swap_dims(df_ints);
  }
  if (BH.rank == 0)
    std::cout << " * Redistributing df integrals." << std::endl;
  MatrixOperations::redistribute(df_ints, BH.Root, BH.Square, true);
  DistributedMatrix::Matrix<double> orbs_t(orbs.ncols, orbs.nrows, BH.Square);
  MatrixOperations::transpose(orbs, orbs_t);
  int msquare = orbs.ncols * orbs.ncols;
  int nbasis = orbs.ncols;
  int naux = df_ints.nrows;
  int nthc = orbs.nrows;
  BxL.setup_matrix(naux, nthc, BH.Square);
  // Form BxL
  if (BH.rank == 0)
    std::cout << " * Redistributing orbs_t." << std::endl;
  // Form T[ik,L]
  MatrixOperations::redistribute(orbs_t, BH.Square, BH.Column, true, nbasis, 1);
  DistributedMatrix::Matrix<double> orb_prod(msquare, orbs.nrows, BH.Column, msquare, 1);
  // T[ik,L] = orbs[i,L] orbs[k,L]
  MatrixOperations::tensor_rank_one(orbs_t, orb_prod);
  MatrixOperations::redistribute(orb_prod, BH.Column, BH.Square);
  MatrixOperations::product(df_ints, orb_prod, BxL);
}

void get_BxL(std::string input_file,
             ContextHandler::BlacsHandler& BH,
             DistributedMatrix::Matrix<double>& BxL)
{
  DistributedMatrix::Matrix<double> BxL_(input_file, "BxL", BH.Root, true, false);
  if (BH.rank == 0) {
    // Need to transpose data for scalapack which expects fortran order
    MatrixOperations::local_transpose(BxL_);
    MatrixOperations::swap_dims(BxL_);
  }
  MatrixOperations::redistribute(BxL_, BH.Root, BH.Square, true);
  BxL.setup_matrix(BxL_.nrows, BxL_.ncols, BH.Square);
  std::copy_n(BxL_.store.data(), BxL_.store.size(), BxL.store.data());
}

void find_pseudo_inverse(DistributedMatrix::Matrix<double>& orbs,
                         ContextHandler::BlacsHandler& BH,
                         DistributedMatrix::Matrix<double>& Sinv)
{
  int nthc = orbs.nrows;
  int norb = orbs.ncols;
  DistributedMatrix::Matrix<double> S(nthc, nthc, BH.Square);
  MatrixOperations::product(orbs, orbs, S, 'N', 'T');
  MatrixOperations::hadamard_product(S);
  MatrixOperations::pseudo_inverse(S, Sinv, 1e-12, BH);
}

void find_pseudo_inverse_half(DistributedMatrix::Matrix<double>& orbs_occ,
                              DistributedMatrix::Matrix<double>& orbs_virt,
                              ContextHandler::BlacsHandler& BH,
                              DistributedMatrix::Matrix<double>& Sinv)
{
  int nthc = orbs.nrows;
  int norb = orbs.ncols;
  DistributedMatrix::Matrix<double> Socc(nthc, nthc, BH.Square);
  DistributedMatrix::Matrix<double> Svirt(nthc, nthc, BH.Square);
  MatrixOperations::product(orbs_occ, orbs_occ, Socc, 'N', 'T');
  MatrixOperations::product(orbs_virt, orbs_virt, Svirt, 'N', 'T');
  MatrixOperations::hadamard_product(S, Svirt);
  MatrixOperations::pseudo_inverse(S, Sinv, 1e-12, BH);
}
