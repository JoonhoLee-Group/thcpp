#ifndef MOLECULAR_H
#define MOLECULAR_H

#include "distributed_matrix.h"

void generate_qmcpack_molecular_thc(std::string input_file,
                                    std::string output_file,
                                    bool need_density_fitting,
                                    bool verbose);
void construct_BxL(std::string input_file,
                   DistributedMatrix::Matrix<double>& orbs,
                   ContextHandler::BlacsHandler& BH,
                   DistributedMatrix::Matrix<double>& BxL);
void get_BxL(std::string input_file,
             ContextHandler::BlacsHandler& BH,
             DistributedMatrix::Matrix<double>& BxL);
void find_pseudo_inverse(DistributedMatrix::Matrix<double>& orbs,
                         ContextHandler::BlacsHandler& BH,
                         DistributedMatrix::Matrix<double>& Sinv);
void write_qmcpack_hdf5(std::string output_file,
                        DistributedMatrix::Matrix<double>& hcore,
                        std::vector<int> nelec,
                        std::vector<double> enuc,
                        DistributedMatrix::Matrix<double>& orbs,
                        DistributedMatrix::Matrix<double>& orbs_half,
                        DistributedMatrix::Matrix<double>& luv,
                        DistributedMatrix::Matrix<double>& muv_half,
                        ContextHandler::BlacsHandler BH);
#endif
