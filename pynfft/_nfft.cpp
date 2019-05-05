#include <vector>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "_util.hpp"

extern "C" {
  #include "nfft3.h"
  #include "fftw3.h"
}

namespace py = pybind11;


struct _NFFT {

  // Constructor
  _NFFT(int d,
        py::tuple N,
        int M,
        py::tuple n,
        int m,
        unsigned int nfft_flags,
        unsigned int fftw_flags
        )
    : _N(N), _M(M) {
    assert(d > 0);
    assert(nfft_flags & MALLOC_X);
    assert(nfft_flags & MALLOC_F);
    assert(nfft_flags & MALLOC_F_HAT);
    std::vector<int> N_(d), n_(d);
    for(int i = 0; i < d; ++i) {
      N_[i] = py::cast<int>(N[i]);
      n_[i] = py::cast<int>(n[i]);
    }
    nfft_init_guru(&_c_nfft_plan, d, N_.data(), M, n_.data(), m, nfft_flags, fftw_flags);
  }

  // Property f_hat
  py::array_t<std::complex<double>> f_hat() const {
    // f_hat has shape N and contiguous strides according to double type
    std::vector<ssize_t> shape = as_size_vector(_N);
    std::vector<ssize_t> strides = contig_strides<std::complex<double>>(_N);
    py::array_t<std::complex<double>> arr(shape, strides, reinterpret_cast<std::complex<double> *>(_c_nfft_plan.f_hat));
    return arr;
  }

  // Property f
  py::array_t<std::complex<double>> f() const {
    // f has shape (M,) and contiguous strides (sizeof(double),)
    std::vector<ssize_t> shape { _M };
    std::vector<ssize_t> strides { sizeof(std::complex<double>) };
    py::array_t<std::complex<double>> arr(shape, strides, reinterpret_cast<std::complex<double> *>(_c_nfft_plan.f));
    return arr;
  }

  // Property x
  py::array_t<double> x() const {
    ssize_t d = py::len(_N);
    // x has shape (M, d) and contiguous strides according to double type
    std::vector<ssize_t> shape { _M, d };
    std::vector<ssize_t> strides = contig_strides<double>(py::make_tuple(_M, d));
    py::array_t<double> arr(shape, strides, reinterpret_cast<double *>(_c_nfft_plan.x));
    return arr;
  }

  // Members
  nfft_plan _c_nfft_plan;
  py::tuple _N;
  int _M;
};


PYBIND11_MODULE(_nfft, m) {
  py::class_<_NFFT>(m, "_NFFT")
    .def(py::init<int, py::tuple, int, py::tuple, int, unsigned int, unsigned int>())
    .def("f_hat", &_NFFT::f_hat)
    .def("f", &_NFFT::f)
    .def("x", &_NFFT::x);
}

