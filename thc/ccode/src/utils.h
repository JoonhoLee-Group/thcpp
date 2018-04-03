#ifndef UTILS_H
#define UTILS_H

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

}
#endif
