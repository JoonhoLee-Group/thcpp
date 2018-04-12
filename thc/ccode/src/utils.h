#ifndef UTILS_H
#define UTILS_H

#include "gitsha1.h"
#include <string>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include "json.hpp"

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
  std::cout << "///////////    //       //        //////       //               //       " << std::endl;
  std::cout << "    //         //       //      //             //               //       " << std::endl;
  std::cout << "    //         //       //    //               //               //       " << std::endl;
  std::cout << "    //         ///////////    //         //////////////   ////////////// " << std::endl;
  std::cout << "    //         //       //    //               //               //       " << std::endl;
  std::cout << "    //         //       //     //              //               //       " << std::endl;
  std::cout << "    //         //       //      ///////        //               //       " << std::endl;
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

}
#endif
