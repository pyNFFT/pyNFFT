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

template <typename FLOAT_T> FLOAT_T *alloc_real(size_t num_el);
template <> float *alloc_real<float>(size_t num_el) {
  return fftwf_alloc_real(num_el);
}
template <> double *alloc_real<double>(size_t num_el) {
  return fftw_alloc_real(num_el);
}
template <> long double *alloc_real<long double>(size_t num_el) {
  return fftwl_alloc_real(num_el);
}

template <typename FLOAT_T> std::complex<FLOAT_T> *alloc_complex(size_t num_el);
template <> std::complex<float> *alloc_complex<float>(size_t num_el) {
  return reinterpret_cast<std::complex<float> *>(fftwf_alloc_complex(num_el));
}
template <> std::complex<double> *alloc_complex<double>(size_t num_el) {
  return reinterpret_cast<std::complex<double> *>(fftw_alloc_complex(num_el));
}
template <>
std::complex<long double> *alloc_complex<long double>(size_t num_el) {
  return reinterpret_cast<std::complex<long double> *>(
      fftwl_alloc_complex(num_el));
}

template <typename FLOAT_T> void dealloc(void *fftw_arr);
template <> void dealloc<float>(void *fftw_arr) { fftwf_free(fftw_arr); }
template <> void dealloc<double>(void *fftw_arr) { fftw_free(fftw_arr); }
template <> void dealloc<long double>(void *fftw_arr) { fftwl_free(fftw_arr); }

//
// class _NFFT
//
// A wrapper class around the C NFFT plan struct, exposing analogous properties.
//

template <typename FLOAT_T> class _NFFT {
public:
  _NFFT(py::tuple N,
        int M,
        py::tuple n,
        int m,
        unsigned int nfft_flags,
        unsigned int fftw_flags) {

    // Convert Py collections and perform sanity checks
    int d = py::len(N);
    std::vector<int> N_vec = shape_vec<int>(N);
    std::vector<int> n_vec = shape_vec<int>(n);
    assert(N_vec.size() == n_vec.size());
    ssize_t N_total = total_size<ssize_t>(N);
    // Remove `MALLOC_*` flags from `nfft_flags` to prevent NFFT from allocating
    // that memory automatically. We need to do that ourselves to be able to
    // feed the pointer into the refcounting machinery.
    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));

    // Initialize C plan
    _c_plan = _new_c_plan(N_vec, M, n_vec, m, nfft_flags, fftw_flags);

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

    // `f_hat` has shape `N` and complex dtype
    std::complex<FLOAT_T> *raw_f_hat = alloc_complex(N_total);
    assert(raw_f_hat != nullptr);
    set_f_hat(raw_f_hat);
    py::capsule f_hat_caps(static_cast<void *>(raw_f_hat), _dealloc<FLOAT_T>);
    std::vector<ssize_t> shape_f_hat(N_vec.begin(), N_vec.end());
    _f_hat =
        py::array_t<std::complex<FLOAT_T>>(shape_f_hat, raw_f_hat, f_hat_caps);

    // `f` has shape `(M,)` and complex dtype
    std::complex<FLOAT_T> *raw_f = alloc_complex(static_cast<size_t>(M));
    assert(raw_f != nullptr);
    set_f(raw_f);
    py::capsule f_caps(static_cast<void *>(raw_f), _dealloc<FLOAT_T>);
    std::vector<ssize_t> shape_f{M};
    _f = py::array_t<std::complex<FLOAT_T>>(shape_f, raw_f, f_caps);

    // `x` has shape `(M, d)` and real dtype
    FLOAT_T *raw_x = alloc_real(static_cast<size_t>(M * d));
    assert(raw_x != nullptr);
    set_x(raw_x);
    py::capsule x_caps(static_cast<void *>(raw_x), _dealloc<FLOAT_T>);
    std::vector<ssize_t> shape_x{M, d};
    _x = py::array_t<FLOAT_T>(shape_x, raw_x, x_caps);
  }

  // Virtual member functions, to be specialized for particular FLOAT_T

  virtual ~_NFFT();
  virtual void *_new_c_plan(std::vector<int> &N,
                            int M,
                            std::vector<int> &n,
                            int m,
                            unsigned int nfft_flags,
                            unsigned int fftw_flags) override;

  virtual void set_f_hat(void *new_f_hat) override;
  virtual void set_f(void *new_f) override;
  virtual void set_x(void *new_x) override;

  virtual void _trafo() override;
  virtual void _trafo_direct() override;
  virtual void _adjoint() override;
  virtual void _adjoint_direct() override;

  // Python interface relevant functions

  // Property f_hat
  py::array_t<std::complex<FLOAT_T>> f_hat() const { return _f_hat; }

  // Property f
  py::array_t<std::complex<FLOAT_T>> f() const { return _f; }

  // Property x
  py::array_t<FLOAT_T> x() const { return _x; }

  // Forward trafo
  void trafo(bool use_dft) {
    if (use_dft) {
      _trafo_direct();
    } else {
      _trafo();
    }
  }

  // Adjoint trafo
  void adjoint(bool use_dft) {
    if (use_dft) {
      _adjoint_direct();
    } else {
      _adjoint();
    }
  }

  // Members
  void *_c_plan;
  py::array_t<std::complex<FLOAT_T>> _f_hat;
  py::array_t<std::complex<FLOAT_T>> _f;
  py::array_t<FLOAT_T> _x;
};

// Floating-point-type-dependent functions

template <> class _NFFT<float> {
public:
  // Resource management of the C plan struct

  ~_NFFT() { nfftf_finalize(static_cast<nfftf_plan *>(_c_plan)); };

  void *_new_c_plan(std::vector<int> &N,
                    int M,
                    std::vector<int> &n,
                    int m,
                    unsigned int nfft_flags,
                    unsigned int fftw_flags) {

    nfft_plan pl;

    int d = static_cast<int>(N.size());

    // Initialize plan
    nfftf_init_guru(&pl, d, N.data(), M, n.data(), m, nfft_flags, fftw_flags);

    return static_cast<void *>(&pl);
  }

  // Array memory management (SIMD-aligned alloc and dealloc)
  // Assignment of C plan members

  void set_f_hat(void *new_f_hat) {
    _c_plan->f_hat = reinterpret_cast<fftwf_complex *>(new_f_hat);
  }
  void set_f(void *new_f) {
    _c_plan->f = reinterpret_cast<fftwf_complex *>(new_f);
  }
  void set_x(void *new_x) { _c_plan->x = reinterpret_cast<float *>(new_x); }

  // Forward trafo
  // TODO: GIL?

  void trafo() { nfftf_trafo(_c_plan); }
  void trafo_direct() { nfftf_trafo_direct(_c_plan); }

  // Adjoint trafo
  // TODO: GIL?

  void adjoint() { nfftf_adjoint(_c_plan); }
  void adjoint_direct() { nfftf_adjoint_direct(_c_plan); }
};

template <> class _NFFT<double> {
public:
  void _init_c_plan(std::vector<int> &N,
                    int M,
                    std::vector<int> &n,
                    int m,
                    unsigned int nfft_flags,
                    unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());
    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));
    nfft_init_guru(static_cast<nfft_plan *>(_c_plan),
                   d,
                   N.data(),
                   M,
                   n.data(),
                   m,
                   nfft_flags,
                   fftw_flags);
  }

  ~_NFFT() { nfft_finalize(static_cast<nfft_plan *>(_c_plan)); };

  void set_f_hat(void *new_f_hat) {
    _c_plan->f_hat = reinterpret_cast<fftw_complex *>(new_f_hat);
  }
  void set_f(void *new_f) {
    _c_plan->f = reinterpret_cast<fftw_complex *>(new_f);
  }
  void set_x(void *new_x) { _c_plan->x = reinterpret_cast<float *>(new_x); }

  void trafo() { nfft_trafo(_c_plan); }
  void trafo_direct() { nfft_trafo_direct(_c_plan); }

  void adjoint() { nfft_adjoint(_c_plan); }
  void adjoint_direct() { nfft_adjoint_direct(_c_plan); }
};

template <> class _NFFT<long double> {
public:
  void _init_c_plan(std::vector<int> &N,
                    int M,
                    std::vector<int> &n,
                    int m,
                    unsigned int nfft_flags,
                    unsigned int fftw_flags) {

    assert(N.size() == n.size());
    int d = static_cast<int>(N.size());
    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));
    nfftl_init_guru(static_cast<nfftl_plan *>(_c_plan),
                    d,
                    N.data(),
                    M,
                    n.data(),
                    m,
                    nfft_flags,
                    fftw_flags);
  }

  ~_NFFT() { nfftf_finalize(static_cast<nfftf_plan *>(_c_plan)); };

  void set_f_hat(void *new_f_hat) {
    _c_plan->f_hat = reinterpret_cast<fftwl_complex *>(new_f_hat);
  }
  void set_f(void *new_f) {
    _c_plan->f = reinterpret_cast<fftwl_complex *>(new_f);
  }
  void set_x(void *new_x) { _c_plan->x = reinterpret_cast<float *>(new_x); }

  void trafo() { nfftl_trafo(_c_plan); }
  void trafo_direct() { nfftl_trafo_direct(_c_plan); }

  void adjoint() { nfftl_adjoint(_c_plan); }
  void adjoint_direct() { nfftl_adjoint_direct(_c_plan); }
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
