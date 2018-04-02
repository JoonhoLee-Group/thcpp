#ifndef UTILS_H
#define UTILS_H

namespace UTILS
{

inline double get_memory(std::vector<double> &vec)
{
  return vec.size() * (double)sizeof(double) / 1000 / 1000 / 1000;
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
