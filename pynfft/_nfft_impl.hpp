#ifndef _NFFT_IMPL_HPP
#define _NFFT_IMPL_HPP

#include <cassert>
#include <iostream>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

extern "C" {
#include "fftw3.h"
#include "nfft3.h"
}

#include "_util.hpp"

namespace py = pybind11;

// Wrapper class for NFFT plan, to be used as templated attribute

template <typename FLOAT_T> struct _NFFTPlan {};

template <> struct _NFFTPlan<double> {
  nfft_plan c_plan;
  // Needed for lifetime management of the stolen reference to the plan arrays;
  // To be used in the array creation routine as `base` parameter.
  py::object refobj;
};

template <> struct _NFFTPlan<float> {
  nfftf_plan c_plan;
  py::object refobj;
};

template <> struct _NFFTPlan<long double> {
  nfftl_plan c_plan;
  py::object refobj;
};

// Thin wrapper class around NFFT, exposing analogous properties

template <typename FLOAT_T> struct _NFFT {

  // Constructor
  _NFFT(int d, py::tuple N, int M, py::tuple n, int m, unsigned int nfft_flags,
        unsigned int fftw_flags);

  // Destructor
  virtual ~_NFFT();

  // Property f_hat
  py::array_t<std::complex<FLOAT_T>> f_hat() const {
    // f_hat has shape N and contiguous strides according to FLOAT_T type
    std::vector<ssize_t> shape = as_ssize_t_vector(_N);
    py::array_t<std::complex<FLOAT_T>> arr(
        shape,
        reinterpret_cast<const std::complex<FLOAT_T> *>(_plan.c_plan.f_hat),
        _plan.refobj);
    std::cout << arr.data() << std::hex << std::endl;
    return arr;
  }

  // Property f
  py::array_t<std::complex<FLOAT_T>> f() const {
    // f has shape (M,) and contiguous strides (sizeof(FLOAT_T),)
    std::vector<ssize_t> shape{_M};
    py::array_t<std::complex<FLOAT_T>> arr(
        shape, reinterpret_cast<const std::complex<FLOAT_T> *>(_plan.c_plan.f),
        _plan.refobj);
    std::cout << arr.data() << std::hex << std::endl;
    return arr;
  }

  // Property x
  py::array_t<FLOAT_T> x() const {
    ssize_t d = py::len(_N);
    // x has shape (M, d) and contiguous strides according to FLOAT_T type
    std::vector<ssize_t> shape{_M, d};
    py::array_t<FLOAT_T> arr(
        shape, reinterpret_cast<const FLOAT_T *>(_plan.c_plan.x), _plan.refobj);
    std::cout << arr.data() << std::hex << std::endl;
    return arr;
  }

  // Members
  py::tuple _N;
  int _M;
  _NFFTPlan<FLOAT_T> _plan;
};

// Specializations of constructor and destructor

template <>
_NFFT<float>::_NFFT(int d, py::tuple N, int M, py::tuple n, int m,
                    unsigned int nfft_flags, unsigned int fftw_flags)
    : _N(N), _M(M) {
  assert(d > 0);
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);
  std::vector<int> N_(d), n_(d);
  for (int i = 0; i < d; ++i) {
    N_[i] = py::cast<int>(N[i]);
    n_[i] = py::cast<int>(n[i]);
  }
  nfftf_init_guru(&_plan.c_plan, d, N_.data(), M, n_.data(), m, nfft_flags,
                  fftw_flags);
  std::cout << "f_hat: " << reinterpret_cast<void *>(_plan.c_plan.f_hat)
            << std::endl
            << "f: " << reinterpret_cast<void *>(_plan.c_plan.f) << std::endl
            << "x: " << reinterpret_cast<void *>(_plan.c_plan.x) << std::endl
            << std::endl;
}

template <> _NFFT<float>::~_NFFT() { nfftf_finalize(&_plan.c_plan); }

template <>
_NFFT<double>::_NFFT(int d, py::tuple N, int M, py::tuple n, int m,
                     unsigned int nfft_flags, unsigned int fftw_flags)
    : _N(N), _M(M) {
  assert(d > 0);
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);
  std::vector<int> N_(d), n_(d);
  for (int i = 0; i < d; ++i) {
    N_[i] = py::cast<int>(N[i]);
    n_[i] = py::cast<int>(n[i]);
  }
  nfft_init_guru(&_plan.c_plan, d, N_.data(), M, n_.data(), m, nfft_flags,
                 fftw_flags);
  std::cout << "f_hat: " << reinterpret_cast<void *>(_plan.c_plan.f_hat)
            << std::endl
            << "f: " << reinterpret_cast<void *>(_plan.c_plan.f) << std::endl
            << "x: " << reinterpret_cast<void *>(_plan.c_plan.x) << std::endl
            << std::endl;
}

template <> _NFFT<double>::~_NFFT() { nfft_finalize(&_plan.c_plan); }

template <>
_NFFT<long double>::_NFFT(int d, py::tuple N, int M, py::tuple n, int m,
                          unsigned int nfft_flags, unsigned int fftw_flags)
    : _N(N), _M(M) {
  assert(d > 0);
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);
  std::vector<int> N_(d), n_(d);
  for (int i = 0; i < d; ++i) {
    N_[i] = py::cast<int>(N[i]);
    n_[i] = py::cast<int>(n[i]);
  }
  nfftl_init_guru(&_plan.c_plan, d, N_.data(), M, n_.data(), m, nfft_flags,
                  fftw_flags);
  std::cout << "f_hat: " << reinterpret_cast<void *>(_plan.c_plan.f_hat)
            << std::endl
            << "f: " << reinterpret_cast<void *>(_plan.c_plan.f) << std::endl
            << "x: " << reinterpret_cast<void *>(_plan.c_plan.x) << std::endl
            << std::endl;
}

template <> _NFFT<long double>::~_NFFT() { nfftl_finalize(&_plan.c_plan); }

// Module-level startup and teardown functions

template <typename FLOAT_T> void _nfft_atentry();

template <> void _nfft_atentry<float>() { fftwf_init_threads(); }

template <> void _nfft_atentry<double>() { fftw_init_threads(); }

template <> void _nfft_atentry<long double>() { fftwl_init_threads(); }

template <typename FLOAT_T> void _nfft_atexit();

template <> void _nfft_atexit<float>() {
  fftwf_cleanup();
  fftwf_cleanup_threads();
}

template <> void _nfft_atexit<double>() {
  fftw_cleanup();
  fftw_cleanup_threads();
}

template <> void _nfft_atexit<long double>() {
  fftwl_cleanup();
  fftwl_cleanup_threads();
}

#endif // _NFFT_IMPL_HPP
