#if not defined _UTIL_HPP
#define _UTIL_HPP

#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;


template <typename T>
std::vector<ssize_t> contig_strides(py::tuple shape) {
  // Return a strides for a C-contiguous array with shape
  // equal to `shape`

  ssize_t d = py::len(shape);
  std::vector<ssize_t> strides(d);

  ssize_t el_stride = 1;
  strides[d - 1] = el_stride * sizeof(T);
  for(ssize_t i = d - 1; i > 0; --i) {
    el_stride *= py::cast<ssize_t>(shape[i]);
    strides[i - 1] = el_stride * sizeof(T);
  }
  return strides;
}


std::vector<ssize_t> as_size_vector(py::tuple N) {
  // Convert tuple of positive integers to a vector of sizes

  ssize_t d = py::len(N);
  std::vector<ssize_t> vec(d);
  for(ssize_t i = 0; i < d; ++i) {
    vec[i] = py::cast<ssize_t>(N[i]);
  }
  return vec;
}

#endif  // _UTIL_HPP
