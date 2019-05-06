#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include "_nfft_impl.hpp"
#include "_util.hpp"

namespace py = pybind11;


PYBIND11_MODULE(_nfft, m) {
  py::class_<_NFFT<float>>(m, "_NFFTFloat")
    .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int, unsigned int>())
    .def_property_readonly("f_hat", &_NFFT<float>::f_hat)
    .def_property_readonly("f", &_NFFT<float>::f)
    .def_property_readonly("x", &_NFFT<float>::x);

  py::class_<_NFFT<double>>(m, "_NFFTDouble")
    .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int, unsigned int>())
    .def_property_readonly("f_hat", &_NFFT<double>::f_hat)
    .def_property_readonly("f", &_NFFT<double>::f)
    .def_property_readonly("x", &_NFFT<double>::x);

  py::class_<_NFFT<long double>>(m, "_NFFTLongDouble")
    .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int, unsigned int>())
    .def_property_readonly("f_hat", &_NFFT<long double>::f_hat)
    .def_property_readonly("f", &_NFFT<long double>::f)
    .def_property_readonly("x", &_NFFT<long double>::x);
}

