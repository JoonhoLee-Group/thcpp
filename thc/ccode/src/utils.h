#ifndef UTILS_H
#define UTILS_H

namespace UTILS
{

inline double get_memory(std::vector<double> &vec)
{
  return vec.size() * (double)sizeof(double) / 1000 / 1000 / 1000;
}

}
#endif
