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

namespace py = pybind11;

// Wrapper class for NFFT plan, mostly to translate from name mangling to
// parametrization via templates
template <typename FLOAT_T> struct _NFFTPlan {};

template <> struct _NFFTPlan<float> {

  // Constructors
  _NFFTPlan() {}
  _NFFTPlan(std::vector<int> N,
            int M,
            std::vector<int> n,
            int m,
            unsigned int nfft_flags,
            unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());

    // Initialize plan internals
    nfftf_init_guru(
        &_c_plan, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);
    // Wrap into Python capsule
    _c_plan_caps = py::capsule(static_cast<void *>(&_c_plan), [](void *ptr) {
      nfftf_finalize(static_cast<nfftf_plan *>(ptr));
    });
  }

  // Forward trafo
  // TODO: GIL?

  void trafo() { nfftf_trafo(&_c_plan); }
  void trafo_direct() { nfftf_trafo_direct(&_c_plan); }

  // Adjoint trafo
  // TODO: GIL?

  void adjoint() { nfftf_adjoint(&_c_plan); }
  void adjoint_direct() { nfftf_adjoint_direct(&_c_plan); }

  // Members

  // C NFFT plan, type depends on floating point data type
  nfftf_plan _c_plan;
  // Python capsule used for lifetime management of the C NFFT plan. It
  // consists of a pointer and a callback for de-allocation that is called once
  // the lifetime of the capsule (or an array object based on it) is over. See
  // https://github.com/pybind/pybind11/issues/1042 for an example.
  py::capsule _c_plan_caps;
};

template <> struct _NFFTPlan<double> {

  _NFFTPlan() {}
  _NFFTPlan(std::vector<int> &N,
            int M,
            std::vector<int> &n,
            int m,
            unsigned int nfft_flags,
            unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());

    // Initialize plan internals
    nfft_init_guru(
        &_c_plan, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);
    // Wrap into Python capsule
    _c_plan_caps = py::capsule(static_cast<void *>(&_c_plan), [](void *ptr) {
      nfft_finalize(static_cast<nfft_plan *>(ptr));
    });
  }

  void trafo() { nfft_trafo(&_c_plan); }
  void trafo_direct() { nfft_trafo_direct(&_c_plan); }
  void adjoint() { nfft_adjoint(&_c_plan); }
  void adjoint_direct() { nfft_adjoint_direct(&_c_plan); }

  nfft_plan _c_plan;
  py::capsule _c_plan_caps;
};

template <> struct _NFFTPlan<long double> {

  _NFFTPlan() {}
  _NFFTPlan(std::vector<int> &N,
            int M,
            std::vector<int> &n,
            int m,
            unsigned int nfft_flags,
            unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());

    // Initialize plan internals
    nfftl_init_guru(
        &_c_plan, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);
    // Wrap into Python capsule
    _c_plan_caps = py::capsule(static_cast<void *>(&_c_plan), [](void *ptr) {
      nfftl_finalize(static_cast<nfftl_plan *>(ptr));
    });
  }

  void trafo() { nfftl_trafo(&_c_plan); }
  void trafo_direct() { nfftl_trafo_direct(&_c_plan); }
  void adjoint() { nfftl_adjoint(&_c_plan); }
  void adjoint_direct() { nfftl_adjoint_direct(&_c_plan); }

  nfftl_plan _c_plan;
  py::capsule _c_plan_caps;
};

// Thin wrapper class around NFFT, exposing analogous properties

template <typename FLOAT_T> struct _NFFT {

  _NFFT(py::tuple N,
        int M,
        py::tuple n,
        int m,
        unsigned int nfft_flags,
        unsigned int fftw_flags) {

    // Basic sanity checks of inputs
    int d = py::len(N);
    assert(d > 0);
    assert(py::len(n) == d);

    // TODO: better to always use `MALLOC_* = false` and allocate ourselves?
    // That would save quite some hassle regarding lifetime management of the C
    // plan. On the other hand, using those flags may not lead to the most
    // tested code paths of the NFFT library...
    assert(nfft_flags & MALLOC_X);
    assert(nfft_flags & MALLOC_F);
    assert(nfft_flags & MALLOC_F_HAT);

    // Convert tuples to int vectors
    std::vector<int> N_vec(d), n_vec(d);
    for (int i = 0; i < d; ++i) {
      N_vec[i] = py::cast<int>(N[i]);
      n_vec[i] = py::cast<int>(n[i]);
    }

    // Initialize plan wrapper
    _plan = _NFFTPlan<FLOAT_T>(N_vec, M, n_vec, m, nfft_flags, fftw_flags);

    // Initialize arrays (i.e., wrap pre-allocated C arrays)
    // NB: this ties the lifetimes of the NFFT plan wrapper (`_c_plan`) and the
    // wrapping arrays closely together.

    // `f_hat` has shape `N` and complex dtype
    std::vector<ssize_t> shape_f_hat(N_vec.begin(), N_vec.end());
    _f_hat = py::array_t<std::complex<FLOAT_T>>(
        shape_f_hat,
        reinterpret_cast<const std::complex<FLOAT_T> *>(_plan._c_plan.f_hat),
        _plan._c_plan_caps);
    // `f` has shape `(M,)` and complex dtype
    std::vector<ssize_t> shape_f{_M};
    _f = py::array_t<std::complex<FLOAT_T>>(
        shape_f,
        reinterpret_cast<const std::complex<FLOAT_T> *>(_plan._c_plan.f),
        _plan._c_plan_caps);
    // `x` has shape `(M, d)` and real dtype
    std::vector<ssize_t> shape_x{_M, d};
    _x =
        py::array_t<FLOAT_T>(shape_x,
                             reinterpret_cast<const FLOAT_T *>(_plan._c_plan.x),
                             _plan._c_plan_caps);
  }

  // Property f_hat
  py::array_t<std::complex<FLOAT_T>> f_hat() const { return _f_hat; }

  // Property f
  py::array_t<std::complex<FLOAT_T>> f() const { return _f; }

  // Property x
  py::array_t<FLOAT_T> x() const { return _x; }

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
  py::array_t<std::complex<FLOAT_T>> _f_hat;
  py::array_t<std::complex<FLOAT_T>> _f;
  py::array_t<FLOAT_T> _x;
};

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
