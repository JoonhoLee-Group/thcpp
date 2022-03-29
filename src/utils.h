#ifndef UTILS_H
#define UTILS_H

#include "gitsha1.h"
#include <string>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include "json.hpp"
#include "mpi.h"

namespace UTILS
{

template <typename T>
inline double get_memory(std::vector<T> &vec)
{
  return vec.size() * double(sizeof(T)) / 1000 / 1000 / 1000;
}

inline std::vector<std::complex< double> > convert_double_to_complex(double *data, int size)
{
  std::vector<std::complex<double> > cdata(size);
  double *d = data;
  for (int i = 0; i < cdata.size(); i++) {
    cdata[i].real(*d);
    cdata[i].imag(0.0);
    d++;
  }
  return cdata;
}

inline std::string get_working_path()
{
  char temp[PATH_MAX];
  return ( getcwd(temp, PATH_MAX) ? std::string( temp ) : std::string("") );
}

inline void print_header(int nprocs, nlohmann::json &input_file)
{
  std::cout << std::endl;
  //std::cout << "######################################################################## " << std::endl;
  std::cout << "///////////    //       //        //////       //               //" << std::endl;
  std::cout << "    //         //       //      //             //               //" << std::endl;
  std::cout << "    //         //       //    //               //               //" << std::endl;
  std::cout << "    //         ///////////    //         //////////////   //////////////" << std::endl;
  std::cout << "    //         //       //    //               //               //" << std::endl;
  std::cout << "    //         //       //     //              //               //" << std::endl;
  std::cout << "    //         //       //      ///////        //               //" << std::endl;
  //std::cout << "######################################################################## " << std::endl;
  std::cout << std::endl;
  std::string short_sha1(g_GIT_SHA1);
  std::string dirty(g_GIT_DIRTY);
  std::string flag;
  if (dirty == "DIRTY") {
    flag = "-dirty";
  } else {
    flag = "";
  }
  auto t = std::chrono::system_clock::now();
  std::time_t running_time = std::chrono::system_clock::to_time_t(t);
  //std::cout << "################################################# " << std::endl;
  std::cout << " * Running on " << nprocs <<  " processors." << std::endl;
  std::cout << " * Git info: " << short_sha1.substr(0,8) + flag << std::endl;
  std::cout << " * " << std::ctime(&running_time);
  std::cout << " * Working directory: " << get_working_path() << std::endl;
  std::cout << " * Input file:" << std::endl;
  std::cout << input_file.dump(4) << std::endl;
  //std::cout << "################################################# " << std::endl;
  std::cout << std::endl;
}

inline void parse_simple_opts(
        nlohmann::json &input,
        int rank,
        int &thc_cfac,
        int &thc_half_cfac,
        bool &half_rotated,
        bool &find_interp_vec)
{
  if (rank == 0) {
    thc_cfac = input.at("thc_cfac").get<int>();
    try {
        find_interp_vec = input.at("find_interp_vec").get<bool>();
    }
    catch(nlohmann::json::out_of_range& error) {
        find_interp_vec = true;
    }
    if (find_interp_vec) {
        try {
          half_rotated = input.at("half_rotated").get<bool>();
        }
        catch (nlohmann::json::out_of_range& error) {
          std::cout << " * Performing THC on full orbital basis." << std::endl;
          half_rotated = false;
        }
        if (half_rotated) {
          try {
            thc_half_cfac = input.at("thc_half_cfac").get<int>();
          }
          catch (nlohmann::json::out_of_range& error) {
            std::cout << " * thc_half_cfac not specified." << std::endl;
            std::cout << " * Setting to be the same as thc_cfac." << std::endl;
            thc_half_cfac = thc_cfac;
          }
        }
      }
  }
  MPI_Bcast(&thc_cfac, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&thc_half_cfac, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&half_rotated, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&find_interp_vec, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
template <typename A, typename B>
inline void zip(const std::vector<A> &a, const std::vector<B> &b, std::vector<std::pair<A,B>> &zipped)
{
  for(size_t i=0; i<a.size(); ++i) {
      zipped.push_back(std::make_pair(a[i], b[i]));
  }
}
template <typename A, typename B>
inline void unzip(
    const std::vector<std::pair<A, B>> &zipped,
    std::vector<A> &a,
    std::vector<B> &b)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

// Stolen from https://stackoverflow.com/questions/37368787/c-sort-one-vector-based-on-another-one
template <typename T>
inline void sort_a_from_b(std::vector<T> &a, std::vector<T> &b)
{
  std::vector<std::pair<T,T> > zipped;
  zip(a, b, zipped);
  std::sort(std::begin(zipped), std::end(zipped),
          [&](const std::pair<T,T>& a, const std::pair<T,T>& b)
          {
              return a.second < b.second;
          });
  unzip(zipped, a, b);
}



}
#endif
