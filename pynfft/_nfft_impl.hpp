#ifndef _NFFT_IMPL_HPP
#define _NFFT_IMPL_HPP

#define _unused(x) ((void)(x))

#include <bitset>  // for debug printing
#include <cassert>
#include <iostream>  // for debug printing
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

extern "C" {
#include "fftw3.h"
#include "nfft3.h"
}

namespace py = pybind11;

// Some utilities

template <typename INT_T> INT_T total_size(py::tuple shape) {
  INT_T size = 1;
  for (auto n : shape) {
    size *= py::cast<INT_T>(n);
  }
  return size;
}

template <typename INT_T> std::vector<INT_T> shape_vec(py::tuple shape) {
  std::vector<INT_T> vec;
  for (auto n : shape) {
    vec.push_back(py::cast<INT_T>(n));
  }
  return vec;
}

// Wrapper class for NFFT plan, mostly to translate from name mangling to
// parametrization via templates
template <typename FLOAT_T> struct _NFFTPlan {};

template <> struct _NFFTPlan<float> {

  _NFFTPlan(std::vector<int> N,
            int M,
            std::vector<int> n,
            int m,
            unsigned int nfft_flags,
            unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());

    // Remove `MALLOC_*` flags from `nfft_flags` to prevent NFFT from allocating
    // that memory automatically. We need to do that ourselves to be able to
    // feed the pointer into the refcounting machinery.
    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));

    // Initialize plan
    nfftf_init_guru(
        &_c_plan, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);
  }

  ~_NFFTPlan() { nfftf_finalize(&_c_plan); }

  // Memory management (SIMD-aligned alloc and dealloc)

  static float *alloc_real(size_t num_el) { return fftwf_alloc_real(num_el); }
  static std::complex<float> *alloc_complex(size_t num_el) {
    return reinterpret_cast<std::complex<float> *>(fftwf_alloc_complex(num_el));
  }
  static void dealloc(void *fftw_arr) { fftwf_free(fftw_arr); }

  // Assignment of C plan members
  void set_f_hat(void *new_f_hat) {
    _c_plan.f_hat = reinterpret_cast<fftwf_complex *>(new_f_hat);
  }
  void set_f(void *new_f) {
    _c_plan.f = reinterpret_cast<fftwf_complex *>(new_f);
  }
  void set_x(void *new_x) { _c_plan.x = reinterpret_cast<float *>(new_x); }

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
};

template <> struct _NFFTPlan<double> {

  _NFFTPlan(std::vector<int> &N,
            int M,
            std::vector<int> &n,
            int m,
            unsigned int nfft_flags,
            unsigned int fftw_flags) {

    std::cout << "ctor with 6 args" << std::endl;
    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());

    std::cout << "N: (";
    for (auto Ni : N) {
      std::cout << Ni << ",";
    }
    std::cout << ")" << std::endl;

    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));
    assert(~(nfft_flags & MALLOC_X));
    assert(~(nfft_flags & MALLOC_F));
    assert(~(nfft_flags & MALLOC_F_HAT));
    nfft_init_guru(
        &_c_plan, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);
    std::cout << "init_guru called" << std::endl;
    std::cout << "plan.N: (";
    for (size_t i = 0; i < N.size(); ++i) {
      std::cout << _c_plan.N[i] << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "plan.N_total: " << _c_plan.N_total << std::endl;
  }

  ~_NFFTPlan() {
    std::cout << "destroy plan" << std::endl;
    nfft_finalize(&_c_plan);
  }

  static double *alloc_real(size_t num_el) { return fftw_alloc_real(num_el); }
  static std::complex<double> *alloc_complex(size_t num_el) {
    return reinterpret_cast<std::complex<double> *>(fftw_alloc_complex(num_el));
  }
  static void dealloc(void *fftw_arr) { fftw_free(fftw_arr); }

  void set_f_hat(void *new_f_hat) {
    _c_plan.f_hat = reinterpret_cast<fftw_complex *>(new_f_hat);
  }
  void set_f(void *new_f) {
    _c_plan.f = reinterpret_cast<fftw_complex *>(new_f);
  }
  void set_x(void *new_x) { _c_plan.x = reinterpret_cast<double *>(new_x); }

  void trafo() {
    std::cout << "calling trafo" << std::endl;
    std::cout << "plan internals:" << std::endl;
    std::cout << "N: (";
    for (size_t i = 0; i < _c_plan.d; ++i) {
      std::cout << _c_plan.N[i] << ",";
    }
    std::cout << ")" << std::endl;
    std::cout << "N_total: " << _c_plan.N_total << std::endl;
    std::cout << "M_total: " << static_cast<size_t>(_c_plan.M_total)
              << std::endl;
    std::cout << "d: " << _c_plan.d << std::endl;
    std::cout << "n_total: " << _c_plan.n_total << std::endl;
    std::cout << "f addr: ";
    std::cout << std::hex << _c_plan.f << std::endl;
    std::cout << "f_hat addr: ";
    std::cout << std::hex << _c_plan.f_hat << std::endl;
    std::cout << "x addr: ";
    std::cout << std::hex << _c_plan.x << std::endl;
    std::bitset<64> bflags(_c_plan.flags);
    std::cout << "flags: " << bflags << std::endl;
    std::bitset<64> bfftw_flags(_c_plan.fftw_flags);
    std::cout << "fftw_flags: " << bfftw_flags << std::endl;
    nfft_trafo(&_c_plan);
  }
  void trafo_direct() { nfft_trafo_direct(&_c_plan); }
  void adjoint() { nfft_adjoint(&_c_plan); }
  void adjoint_direct() { nfft_adjoint_direct(&_c_plan); }

  nfft_plan _c_plan;
};

template <> struct _NFFTPlan<long double> {

  _NFFTPlan(std::vector<int> &N,
            int M,
            std::vector<int> &n,
            int m,
            unsigned int nfft_flags,
            unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());

    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));
    nfftl_init_guru(
        &_c_plan, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);
  }

  ~_NFFTPlan() { nfftl_finalize(&_c_plan); }

  static long double *alloc_real(size_t num_el) { return fftwl_alloc_real(num_el); }
  static std::complex<long double> *alloc_complex(size_t num_el) {
    return reinterpret_cast<std::complex<long double> *>(
        fftwl_alloc_complex(num_el));
  }
  static void dealloc(void *fftw_arr) { fftwl_free(fftw_arr); }

  void set_f_hat(void *new_f_hat) {
    _c_plan.f_hat = reinterpret_cast<fftwl_complex *>(new_f_hat);
  }
  void set_f(void *new_f) {
    _c_plan.f = reinterpret_cast<fftwl_complex *>(new_f);
  }
  void set_x(void *new_x) {
    _c_plan.x = reinterpret_cast<long double *>(new_x);
  }

  void trafo() { nfftl_trafo(&_c_plan); }
  void trafo_direct() { nfftl_trafo_direct(&_c_plan); }
  void adjoint() { nfftl_adjoint(&_c_plan); }
  void adjoint_direct() { nfftl_adjoint_direct(&_c_plan); }

  nfftl_plan _c_plan;
};

// A factory function for `_NFFTPlan`, to be used in the initializer of `_NFFT`

template <typename FLOAT_T>
_NFFTPlan<FLOAT_T> make_NFFTPlan(py::tuple N,
                                 int M,
                                 py::tuple n,
                                 int m,
                                 unsigned int nfft_flags,
                                 unsigned int fftw_flags) {
  // Basic sanity checks and conversions
  int d = py::len(N);
  assert(d > 0);
  assert(py::len(n) == d);
  _unused(d);
  // TODO: allow these to be off for lazy allocation or even user-provided
  // memory?
  assert(nfft_flags & MALLOC_X);
  assert(nfft_flags & MALLOC_F);
  assert(nfft_flags & MALLOC_F_HAT);
  // `_NFFTPlan` needs `std::vector`s for shapes
  std::vector<int> N_vec = shape_vec<int>(N);
  std::vector<int> n_vec = shape_vec<int>(n);

  std::cout << "N_vec: ";
  for (auto Ni : N_vec) {
    std::cout << Ni << " " << std::endl;
  }

  // Create plan wrapper and return it
  std::cout << "create C plan wrapper" << std::endl;
  return _NFFTPlan<FLOAT_T>(N_vec, M, n_vec, m, nfft_flags, fftw_flags);
}

// Wrapper class around NFFT, exposing analogous properties

template <typename FLOAT_T> struct _NFFT {

  _NFFT(py::tuple N,
        int M,
        py::tuple n,
        int m,
        unsigned int nfft_flags,
        unsigned int fftw_flags)
      : _plan(make_NFFTPlan<FLOAT_T>(N, M, n, m, nfft_flags, fftw_flags)) {

    {
      std::cout << "After Init" << std::endl;
      std::cout << "plan internals:" << std::endl;
      std::cout << "N: (";
      for (size_t i = 0; i < _plan._c_plan.d; ++i) {
        std::cout << _plan._c_plan.N[i] << ",";
      }
      std::cout << ")" << std::endl;
      std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;
      std::cout << "M_total: " << static_cast<size_t>(_plan._c_plan.M_total)
                << std::endl;
      std::cout << "d: " << _plan._c_plan.d << std::endl;
      std::cout << "n_total: " << _plan._c_plan.n_total << std::endl;
      std::cout << "f addr: ";
      std::cout << std::hex << _plan._c_plan.f << std::endl;
      std::cout << "f_hat addr: ";
      std::cout << std::hex << _plan._c_plan.f_hat << std::endl;
      std::cout << "x addr: ";
      std::cout << std::hex << _plan._c_plan.x << std::endl;
      std::bitset<64> bflags(_plan._c_plan.flags);
      std::cout << "flags: " << bflags << std::endl;
      std::bitset<64> bfftw_flags(_plan._c_plan.fftw_flags);
      std::cout << "fftw_flags: " << bfftw_flags << std::endl;
    }
    // TODO: here the value is already wrong!!
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;

    // Allocate arrays with FFTW's aligned allocator and wrap them as Numpy
    // arrays.
    // NOTE: The general procedure is to first allocate memory using FFTW's
    // SIMD-aligning allocator, then wrap the pointer into a Python capsule for
    // lifetime management, and finally use that capsule as a base in the array
    // creation. That way, deallocation is done automatically at the correct
    // point in time.
    // The pointer is also assigned to the C NFFT plan struct.
    // The capsule is constructed using an array base pointer and a callback
    // function for deallocation of its memory. See
    // https://github.com/pybind/pybind11/issues/1042 for some discussion on the
    // usage of capsules.

    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;
    // Sanity checks have been done in the `_plan` initializer
    int d = py::len(N);
    std::vector<int> N_vec = shape_vec<int>(N);
    std::vector<int> n_vec = shape_vec<int>(n);
    ssize_t N_total = total_size<ssize_t>(N);

    // `f_hat` has shape `N` and complex dtype
    std::cout << "create f_hat" << std::endl;
    std::complex<FLOAT_T> *raw_f_hat = _NFFTPlan<FLOAT_T>::alloc_complex(N_total);
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;
    assert(raw_f_hat != nullptr);
    const_cast<_NFFTPlan<FLOAT_T>&>(_plan).set_f_hat(raw_f_hat);
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;
    std::cout << "f_hat addr after setting: ";
    std::cout << std::hex << _plan._c_plan.f_hat << std::endl;
    std::cout << "raw_f_hat addr: ";
    std::cout << std::hex << raw_f_hat << std::endl;
    py::capsule f_hat_caps(static_cast<void *>(raw_f_hat), _dealloc_array);
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;
    std::vector<ssize_t> shape_f_hat(N_vec.begin(), N_vec.end());
    _f_hat =
        py::array_t<std::complex<FLOAT_T>>(shape_f_hat, raw_f_hat, f_hat_caps);
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;

    // `f` has shape `(M,)` and complex dtype
    std::cout << "create f" << std::endl;
    std::complex<FLOAT_T> *raw_f = _NFFTPlan<FLOAT_T>::alloc_complex(static_cast<size_t>(M));
    assert(raw_f != nullptr);
    const_cast<_NFFTPlan<FLOAT_T>&>(_plan).set_f(raw_f);
    py::capsule f_caps(static_cast<void *>(raw_f), _dealloc_array);
    std::vector<ssize_t> shape_f{M};
    _f = py::array_t<std::complex<FLOAT_T>>(shape_f, raw_f, f_caps);
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;

    // `x` has shape `(M, d)` and real dtype
    std::cout << "create x" << std::endl;
    FLOAT_T *raw_x = _NFFTPlan<FLOAT_T>::alloc_real(static_cast<size_t>(M * d));
    assert(raw_x != nullptr);
    const_cast<_NFFTPlan<FLOAT_T>&>(_plan).set_x(raw_x);  // Cast away const-ness
    py::capsule x_caps(static_cast<void *>(raw_x), _dealloc_array);
    std::vector<ssize_t> shape_x{M, d};
    _x = py::array_t<FLOAT_T>(shape_x, raw_x, x_caps);
    std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;

    {
      std::cout << "plan internals:" << std::endl;
      std::cout << "N: (";
      for (size_t i = 0; i < _plan._c_plan.d; ++i) {
        std::cout << _plan._c_plan.N[i] << ",";
      }
      std::cout << ")" << std::endl;
      std::cout << "N_total: " << _plan._c_plan.N_total << std::endl;
      std::cout << "M_total: " << static_cast<size_t>(_plan._c_plan.M_total)
                << std::endl;
      std::cout << "d: " << _plan._c_plan.d << std::endl;
      std::cout << "n_total: " << _plan._c_plan.n_total << std::endl;
      std::cout << "f addr: ";
      std::cout << std::hex << _plan._c_plan.f << std::endl;
      std::cout << "f_hat addr: ";
      std::cout << std::hex << _plan._c_plan.f_hat << std::endl;
      std::cout << "x addr: ";
      std::cout << std::hex << _plan._c_plan.x << std::endl;
      std::bitset<64> bflags(_plan._c_plan.flags);
      std::cout << "flags: " << bflags << std::endl;
      std::bitset<64> bfftw_flags(_plan._c_plan.fftw_flags);
      std::cout << "fftw_flags: " << bfftw_flags << std::endl;
    }
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
      const_cast<_NFFTPlan<FLOAT_T>&>(_plan).trafo_direct();
    } else {
      const_cast<_NFFTPlan<FLOAT_T>&>(_plan).trafo();
    }
  }

  // Adjoint trafo
  void adjoint(bool use_dft) {
    if (use_dft) {
      const_cast<_NFFTPlan<FLOAT_T>&>(_plan).adjoint_direct();
    } else {
      const_cast<_NFFTPlan<FLOAT_T>&>(_plan).adjoint();
    }
  }

  // Helpers
  static void _dealloc_array(void *ptr) {
    std::cout << "dealloc array" << std::endl;
    _NFFTPlan<FLOAT_T>::dealloc(ptr);
  }

  // Members
  _NFFTPlan<FLOAT_T> const &_plan;
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
