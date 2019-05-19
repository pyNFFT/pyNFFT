#if not defined _UTIL_HPP
#define _UTIL_HPP

#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

std::vector<ssize_t> as_ssize_t_vector(py::tuple N) {
  // Convert tuple of positive integers to a vector of sizes

  ssize_t d = py::len(N);
  std::vector<ssize_t> vec(d);
  for (ssize_t i = 0; i < d; ++i) {
    vec[i] = py::cast<ssize_t>(N[i]);
  }
  return vec;
}

#endif // _UTIL_HPP
