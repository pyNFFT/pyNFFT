// Copyright PyNFFT developers and contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

//
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
