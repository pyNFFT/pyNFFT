#ifndef _NFFT_IMPL_HPP
#define _NFFT_IMPL_HPP

#define _unused(x) ((void)(x))  // to avoid warnings for assert-only variables

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

//
// Utilities
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

//
// Template aliases for NFFT plan structs and FFTW complex types (mapping name
// mangling to template parameter)
//

template <typename FLOAT_T> struct nfft_plan_t_impl {};
template <> struct nfft_plan_t_impl<float> { typedef nfftf_plan TYPE; };
template <> struct nfft_plan_t_impl<double> { typedef nfft_plan TYPE; };
template <> struct nfft_plan_t_impl<long double> { typedef nfftl_plan TYPE; };

template <typename FLOAT_T>
using nfft_plan_t = typename nfft_plan_t_impl<FLOAT_T>::TYPE;

template <typename FLOAT_T> struct fftw_complex_t_impl {};
template <> struct fftw_complex_t_impl<float> { typedef fftwf_complex TYPE; };
template <> struct fftw_complex_t_impl<double> { typedef fftw_complex TYPE; };
template <> struct fftw_complex_t_impl<long double> {
  typedef fftwl_complex TYPE;
};

template <typename FLOAT_T>
using fftw_complex_t = typename fftw_complex_t_impl<FLOAT_T>::TYPE;

//
// Plan initialization and finalization
//

template <typename FLOAT_T>
void nfft_plan_init(nfft_plan_t<FLOAT_T> *plan,
                    int d,
                    int *N,
                    int M,
                    int *n,
                    int m,
                    unsigned nfft_flags,
                    unsigned fftw_flags);
template <>
void nfft_plan_init<float>(nfft_plan_t<float> *plan,
                           int d,
                           int *N,
                           int M,
                           int *n,
                           int m,
                           unsigned nfft_flags,
                           unsigned fftw_flags) {
  nfftf_init_guru(plan, d, N, M, n, m, nfft_flags, fftw_flags);
}
template <>
void nfft_plan_init<double>(nfft_plan_t<double> *plan,
                            int d,
                            int *N,
                            int M,
                            int *n,
                            int m,
                            unsigned nfft_flags,
                            unsigned fftw_flags) {
  nfft_init_guru(plan, d, N, M, n, m, nfft_flags, fftw_flags);
}
template <>
void nfft_plan_init<long double>(nfft_plan_t<long double> *plan,
                                 int d,
                                 int *N,
                                 int M,
                                 int *n,
                                 int m,
                                 unsigned nfft_flags,
                                 unsigned fftw_flags) {
  nfftl_init_guru(plan, d, N, M, n, m, nfft_flags, fftw_flags);
}

template <typename FLOAT_T> void nfft_plan_finalize(nfft_plan_t<FLOAT_T> *plan);
template <> void nfft_plan_finalize<float>(nfft_plan_t<float> *plan) {
  nfftf_finalize(plan);
}
template <> void nfft_plan_finalize<double>(nfft_plan_t<double> *plan) {
  nfft_finalize(plan);
}
template <>
void nfft_plan_finalize<long double>(nfft_plan_t<long double> *plan) {
  nfftl_finalize(plan);
}

//
// SIMD-aligned memory allocation and deallocation
//

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

template <typename FLOAT_T>
fftw_complex_t<FLOAT_T> *alloc_complex(size_t num_el);
template <> fftw_complex_t<float> *alloc_complex<float>(size_t num_el) {
  return fftwf_alloc_complex(num_el);
}
template <> fftw_complex_t<double> *alloc_complex<double>(size_t num_el) {
  return fftw_alloc_complex(num_el);
}
template <>
fftw_complex_t<long double> *alloc_complex<long double>(size_t num_el) {
  return fftwl_alloc_complex(num_el);
}

template <typename FLOAT_T> void dealloc(void *fftw_arr);
template <> void dealloc<float>(void *fftw_arr) { fftwf_free(fftw_arr); }
template <> void dealloc<double>(void *fftw_arr) { fftw_free(fftw_arr); }
template <> void dealloc<long double>(void *fftw_arr) { fftwl_free(fftw_arr); }

//
// Precomputation
//

template <typename FLOAT_T>
void nfft_precompute_one_psi_impl(nfft_plan_t<FLOAT_T> *plan);
template <> void nfft_precompute_one_psi_impl<float>(nfft_plan_t<float> *plan) {
  nfftf_precompute_one_psi(plan);
}
template <>
void nfft_precompute_one_psi_impl<double>(nfft_plan_t<double> *plan) {
  nfft_precompute_one_psi(plan);
}
template <>
void nfft_precompute_one_psi_impl<long double>(nfft_plan_t<long double> *plan) {
  nfftl_precompute_one_psi(plan);
}

//
// Forward and adjoint NFFT implementations
//
// TODO: GIL?
//

template <typename FLOAT_T> void nfft_trafo_impl(nfft_plan_t<FLOAT_T> *plan);
template <> void nfft_trafo_impl<float>(nfft_plan_t<float> *plan) {
  nfftf_trafo(plan);
}
template <> void nfft_trafo_impl<double>(nfft_plan_t<double> *plan) {
  nfft_trafo(plan);
}
template <> void nfft_trafo_impl<long double>(nfft_plan_t<long double> *plan) {
  nfftl_trafo(plan);
}

template <typename FLOAT_T>
void nfft_trafo_direct_impl(nfft_plan_t<FLOAT_T> *plan);
template <> void nfft_trafo_direct_impl<float>(nfft_plan_t<float> *plan) {
  nfftf_trafo_direct(plan);
}
template <> void nfft_trafo_direct_impl<double>(nfft_plan_t<double> *plan) {
  nfft_trafo_direct(plan);
}
template <>
void nfft_trafo_direct_impl<long double>(nfft_plan_t<long double> *plan) {
  nfftl_trafo_direct(plan);
}

template <typename FLOAT_T> void nfft_adjoint_impl(nfft_plan_t<FLOAT_T> *plan);
template <> void nfft_adjoint_impl<float>(nfft_plan_t<float> *plan) {
  nfftf_adjoint(plan);
}
template <> void nfft_adjoint_impl<double>(nfft_plan_t<double> *plan) {
  nfft_adjoint(plan);
}
template <>
void nfft_adjoint_impl<long double>(nfft_plan_t<long double> *plan) {
  nfftl_adjoint(plan);
}

template <typename FLOAT_T>
void nfft_adjoint_direct_impl(nfft_plan_t<FLOAT_T> *plan);
template <> void nfft_adjoint_direct_impl<float>(nfft_plan_t<float> *plan) {
  nfftf_adjoint_direct(plan);
}
template <> void nfft_adjoint_direct_impl<double>(nfft_plan_t<double> *plan) {
  nfft_adjoint_direct(plan);
}
template <>
void nfft_adjoint_direct_impl<long double>(nfft_plan_t<long double> *plan) {
  nfftl_adjoint_direct(plan);
}

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
    size_t N_total = total_size<size_t>(N);
    // Remove `MALLOC_*` flags from `nfft_flags` to prevent NFFT from allocating
    // that memory automatically. We need to do that ourselves to be able to
    // feed the pointer into the refcounting machinery.
    nfft_flags = nfft_flags & (~(MALLOC_X | MALLOC_F | MALLOC_F_HAT));

    // Initialize C plan
    nfft_plan_init<FLOAT_T>(
        &plan_, d, N_vec.data(), M, n_vec.data(), m, nfft_flags, fftw_flags);

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
    plan_.f_hat = alloc_complex<FLOAT_T>(N_total);
    assert(plan_.f_hat != nullptr);
    py::capsule f_hat_caps(static_cast<void *>(plan_.f_hat), dealloc<FLOAT_T>);
    std::vector<ssize_t> f_hat_shape(N_vec.begin(), N_vec.end());
    f_hat_ = py::array_t<std::complex<FLOAT_T>>(
        f_hat_shape,
        reinterpret_cast<std::complex<FLOAT_T> *>(plan_.f_hat),
        f_hat_caps);

    // `f` has shape `(M,)` and complex dtype
    plan_.f = alloc_complex<FLOAT_T>(static_cast<size_t>(M));
    assert(plan_.raw_f != nullptr);
    py::capsule f_caps(static_cast<void *>(plan_.f), dealloc<FLOAT_T>);
    std::vector<ssize_t> f_shape{M};
    f_ = py::array_t<std::complex<FLOAT_T>>(
        f_shape, reinterpret_cast<std::complex<FLOAT_T> *>(plan_.f), f_caps);

    // `x` has shape `(M, d)` and real dtype
    plan_.x = alloc_real<FLOAT_T>(static_cast<size_t>(M * d));
    assert(plan_.x != nullptr);
    py::capsule x_caps(static_cast<void *>(plan_.x), dealloc<FLOAT_T>);
    std::vector<ssize_t> x_shape{M, d};
    x_ = py::array_t<FLOAT_T>(x_shape, plan_.x, x_caps);
  }

  ~_NFFT() { nfft_plan_finalize<FLOAT_T>(&plan_); }

  // Python interface relevant functions

  // Property f_hat
  py::array_t<std::complex<FLOAT_T>> f_hat() const { return f_hat_; }

  // Property f
  py::array_t<std::complex<FLOAT_T>> f() const { return f_; }

  // Property x
  py::array_t<FLOAT_T> x() const { return x_; }

  // Precomputation
  void precompute() { nfft_precompute_one_psi_impl<FLOAT_T>(&plan_); }

  // Forward trafo
  void trafo(bool use_dft) {
    if (use_dft) {
      nfft_trafo_direct_impl<FLOAT_T>(&plan_);
    } else {
      nfft_trafo_impl<FLOAT_T>(&plan_);
    }
  }

  // Adjoint trafo
  void adjoint(bool use_dft) {
    if (use_dft) {
      nfft_adjoint_direct_impl<FLOAT_T>(&plan_);
    } else {
      nfft_adjoint_impl<FLOAT_T>(&plan_);
    }
  }

  // Members
  nfft_plan_t<FLOAT_T> plan_;
  py::array_t<std::complex<FLOAT_T>> f_hat_;
  py::array_t<std::complex<FLOAT_T>> f_;
  py::array_t<FLOAT_T> x_;
};

//
// Module-level startup and teardown functions
//

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
