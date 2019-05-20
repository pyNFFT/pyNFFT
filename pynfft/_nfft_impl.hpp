#ifndef _NFFT_IMPL_HPP
#define _NFFT_IMPL_HPP

#include <cassert>
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

// Wrapper class for NFFT plan, mostly to translate from name mangling to
// parametrization via templates
template <typename FLOAT_T> struct _NFFTPlan {};

template <> struct _NFFTPlan<float> {

  // Constructor

  _NFFTPlan() {
    c_plan_caps = py::capsule(reinterpret_cast<void *>(&c_plan), [](void *ptr) {
      nfftf_finalize(reinterpret_cast<nfftf_plan *>(ptr));
    });
    c_arr_caps = py::capsule(nullptr, [](void *ptr) {});
  }

  // Forward trafo
  // TODO: GIL?

  void trafo() { nfftf_trafo(&c_plan); }
  void trafo_direct() { nfftf_trafo_direct(&c_plan); }

  // Adjoint trafo
  // TODO: GIL?

  void adjoint() { nfftf_adjoint(&c_plan); }
  void adjoint_direct() { nfftf_adjoint_direct(&c_plan); }

  // Members

  // C NFFT plan, type depends on floating point data type
  nfftf_plan c_plan;
  // Python capsule, used for lifetime management of the C NFFT plan
  py::capsule c_plan_caps;
  // Trivial capsules for the C NFFT array members, intended to be used in the
  // array creation routine as `base` parameter. See
  // https://github.com/pybind/pybind11/issues/1042 for a bit of rationale. The
  // lifetime of those arrays should be that of the plan, not of the arrays
  // generated in the wrapper class.
  py::capsule c_arr_caps;
};

template <> struct _NFFTPlan<double> {
  _NFFTPlan() {
    c_plan_caps = py::capsule(reinterpret_cast<void *>(&c_plan), [](void *ptr) {
      nfft_finalize(reinterpret_cast<nfft_plan *>(ptr));
    });
    c_arr_caps = py::capsule(nullptr, [](void *ptr) {});
  }

  void trafo() { nfft_trafo(&c_plan); }
  void trafo_direct() { nfft_trafo_direct(&c_plan); }
  void adjoint() { nfft_adjoint(&c_plan); }
  void adjoint_direct() { nfft_adjoint_direct(&c_plan); }

  nfft_plan c_plan;
  py::capsule c_plan_caps;
  py::capsule c_arr_caps;
};

template <> struct _NFFTPlan<long double> {
  _NFFTPlan() {
    c_plan_caps = py::capsule(reinterpret_cast<void *>(&c_plan), [](void *ptr) {
      nfftl_finalize(reinterpret_cast<nfftl_plan *>(ptr));
    });
    c_arr_caps = py::capsule(nullptr, [](void *ptr) {});
  }

  void trafo() { nfftl_trafo(&c_plan); }
  void trafo_direct() { nfftl_trafo_direct(&c_plan); }
  void adjoint() { nfftl_adjoint(&c_plan); }
  void adjoint_direct() { nfftl_adjoint_direct(&c_plan); }

  nfftl_plan c_plan;
  py::capsule c_plan_caps;
  py::capsule c_arr_caps;
};

// Thin wrapper class around NFFT, exposing analogous properties

template <typename FLOAT_T> struct _NFFT {

  // Constructor
  _NFFT(int d,
        py::tuple N,
        int M,
        py::tuple n,
        int m,
        unsigned int nfft_flags,
        unsigned int fftw_flags);

  // Property f_hat
  py::array_t<std::complex<FLOAT_T>> f_hat() const {
    // f_hat has shape N and contiguous strides according to FLOAT_T type
    std::vector<ssize_t> shape = as_ssize_t_vector(_N);
    // TODO: cache array instead of creating every time
    py::array_t<std::complex<FLOAT_T>> arr(
        shape,
        reinterpret_cast<const std::complex<FLOAT_T> *>(_plan.c_plan.f_hat),
        _plan.c_arr_caps);
    return arr;
  }

  // Property f
  py::array_t<std::complex<FLOAT_T>> f() const {
    // f has shape (M,) and contiguous strides (sizeof(FLOAT_T),)
    std::vector<ssize_t> shape{_M};
    // TODO: cache array instead of creating every time
    py::array_t<std::complex<FLOAT_T>> arr(
        shape,
        reinterpret_cast<const std::complex<FLOAT_T> *>(_plan.c_plan.f),
        _plan.c_arr_caps);
    return arr;
  }

  // Property x
  py::array_t<FLOAT_T> x() const {
    ssize_t d = py::len(_N);
    // x has shape (M, d) and contiguous strides according to FLOAT_T type
    std::vector<ssize_t> shape{_M, d};
    // TODO: cache array instead of creating every time
    py::array_t<FLOAT_T> arr(shape,
                             reinterpret_cast<const FLOAT_T *>(_plan.c_plan.x),
                             _plan.c_arr_caps);
    return arr;
  }

  // Forward trafo
  void trafo(bool use_dft) {
    if (use_dft) {
      _plan.trafo_direct();
    } else {
      _plan.trafo();
    }
  }

  // Adjoint trafo
  void adjoint(bool use_dft) {
    if (use_dft) {
      _plan.adjoint_direct();
    } else {
      _plan.adjoint();
    }
  }

  // Members
  py::tuple _N;
  int _M;
  _NFFTPlan<FLOAT_T> _plan;
};

// Specializations of constructor and destructor

template <>
_NFFT<float>::_NFFT(int d,
                    py::tuple N,
                    int M,
                    py::tuple n,
                    int m,
                    unsigned int nfft_flags,
                    unsigned int fftw_flags)
    : _N(N), _M(M) {

  // Basic sanity checks of inputs
  assert(d > 0);
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);

  std::vector<int> N_(d), n_(d);
  for (int i = 0; i < d; ++i) {
    N_[i] = py::cast<int>(N[i]);
    n_[i] = py::cast<int>(n[i]);
  }
  nfftf_init_guru(
      &_plan.c_plan, d, N_.data(), M, n_.data(), m, nfft_flags, fftw_flags);
}

template <>
_NFFT<double>::_NFFT(int d,
                     py::tuple N,
                     int M,
                     py::tuple n,
                     int m,
                     unsigned int nfft_flags,
                     unsigned int fftw_flags)
    : _N(N), _M(M) {

  // Basic sanity checks of inputs
  assert(d > 0);
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);

  std::vector<int> N_(d), n_(d);
  for (int i = 0; i < d; ++i) {
    N_[i] = py::cast<int>(N[i]);
    n_[i] = py::cast<int>(n[i]);
  }
  nfft_init_guru(
      &_plan.c_plan, d, N_.data(), M, n_.data(), m, nfft_flags, fftw_flags);
}

template <>
_NFFT<long double>::_NFFT(int d,
                          py::tuple N,
                          int M,
                          py::tuple n,
                          int m,
                          unsigned int nfft_flags,
                          unsigned int fftw_flags)
    : _N(N), _M(M) {

  // Basic sanity checks of inputs
  assert(d > 0);
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);

  std::vector<int> N_(d), n_(d);
  for (int i = 0; i < d; ++i) {
    N_[i] = py::cast<int>(N[i]);
    n_[i] = py::cast<int>(n[i]);
  }
  nfftl_init_guru(
      &_plan.c_plan, d, N_.data(), M, n_.data(), m, nfft_flags, fftw_flags);
}

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

#endif  // _NFFT_IMPL_HPP
