// _nfft_adaptor.hpp -- Adaptor types and functions for NFFT
//
// Basically translation of C name mangling to C++ template parameters.
//

#ifndef _NFFT_ADAPTOR_HPP
#define _NFFT_ADAPTOR_HPP

#define _unused(x) ((void)(x))  // to avoid warnings for assert-only variables

#include <bitset>  // for debugging
#include <cassert>

extern "C" {
#include "fftw3.h"
#include "nfft3.h"
}

#include "_util.hpp"

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
// Forward transforms
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

//
// Adjoint transforms
//
// TODO: GIL?
//

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

#endif  // _NFFT_ADAPTOR_HPP
