#ifndef UTILS_H
#define UTILS_H

#include "gitsha1.h"
#include <string>

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

inline void print_header(int nprocs)
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
    flag = "dirty";
  } else {
    flag = "";
  }
  std::cout << "############################################################ " << std::endl;
  std::cout << "# Running on : " << nprocs <<  " processors." << std::endl;
  std::cout << "# Git info: " << short_sha1.substr(0,8) + "-" + flag << std::endl;
  std::cout << "############################################################ " << std::endl;
  std::cout << std::endl;
}

}
#endif
