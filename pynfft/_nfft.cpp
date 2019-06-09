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
// _nfft.cpp -- definition of the `_nfft` Python extension module
//

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include "_nfft_impl.hpp"

namespace py = pybind11;

// NB: The startup functions can just be run as part of the module creation code
// (PYBIND11_MODULE), whereas the teardown functions need to be registered as
// callbacks somewhere. We use the approach with the Python `atexit` module, see
// https://pybind11.readthedocs.io/en/master/advanced/misc.html#module-destructors
// for details.

PYBIND11_MODULE(_nfft, m) {
  m.doc() = "Wrapper module for C NFFT plans and associated functions.";
  auto atexit = py::module::import("atexit");

  py::class_<_NFFT<float>>(m, "_NFFTFloat")
      .def(py::init<py::tuple,
                    int,
                    py::tuple,
                    int,
                    unsigned int,
                    unsigned int>())
      .def_property_readonly("f_hat", &_NFFT<float>::f_hat)
      .def_property_readonly("f", &_NFFT<float>::f)
      .def_property_readonly("x", &_NFFT<float>::x)
      .def("precompute", &_NFFT<float>::precompute)
      .def("trafo", &_NFFT<float>::trafo)
      .def("adjoint", &_NFFT<float>::adjoint);
  _nfft_atentry<float>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<float>));

  py::class_<_NFFT<double>>(m, "_NFFTDouble")
      .def(py::init<py::tuple,
                    int,
                    py::tuple,
                    int,
                    unsigned int,
                    unsigned int>())
      .def_property_readonly("f_hat", &_NFFT<double>::f_hat)
      .def_property_readonly("f", &_NFFT<double>::f)
      .def_property_readonly("x", &_NFFT<double>::x)
      .def("precompute", &_NFFT<double>::precompute)
      .def("trafo", &_NFFT<double>::trafo)
      .def("adjoint", &_NFFT<double>::adjoint);
  _nfft_atentry<double>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<double>));

  py::class_<_NFFT<long double>>(m, "_NFFTLongDouble")
      .def(py::init<py::tuple,
                    int,
                    py::tuple,
                    int,
                    unsigned int,
                    unsigned int>())
      .def_property_readonly("f_hat", &_NFFT<long double>::f_hat)
      .def_property_readonly("f", &_NFFT<long double>::f)
      .def_property_readonly("x", &_NFFT<long double>::x)
      .def("precompute", &_NFFT<long double>::precompute)
      .def("trafo", &_NFFT<long double>::trafo)
      .def("adjoint", &_NFFT<long double>::adjoint);
  _nfft_atentry<long double>();
  atexit.attr("register")(py::cpp_function(_nfft_atexit<long double>));
}
