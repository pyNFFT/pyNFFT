// _util.hpp -- Utility code
//

#ifndef _UTIL_HPP
#define _UTIL_HPP

#define _unused(x) ((void)(x))  // to avoid warnings for assert-only variables

#include <iostream>  // for debugging
#include <type_traits>
#include <vector>

#include <pybind11/pybind11.h>

extern "C" {
#include "fftw3.h"
#include "nfft3.h"
}

#include "_util.hpp"

namespace py = pybind11;

//
// Conversion from Python
//

template <typename INT_T> INT_T total_size(py::tuple shape) {
  static_assert(std::is_integral<INT_T>::value, "integral type needed");
  INT_T size = 1;
  for (auto n : shape) {
    size *= py::cast<INT_T>(n);
  }
  return size;
}

template <typename INT_T> std::vector<INT_T> shape_vec(py::tuple shape) {
  static_assert(std::is_integral<INT_T>::value, "integral type needed");
  std::vector<INT_T> vec;
  for (auto n : shape) {
    vec.push_back(py::cast<INT_T>(n));
  }
  return vec;
}

#endif  // _UTIL_HPP
