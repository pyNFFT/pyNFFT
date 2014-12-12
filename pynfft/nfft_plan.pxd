# -*- coding: utf-8 -*-
#
# Copyright (c) 2013, 2014 Ghislain Antony Vaillant
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from mv_plan cimport *
from cnfft3 cimport (nfft_plan, fftw_complex)
from cnfft3 cimport (nfft_malloc, nfft_finalize, nfft_free, nfft_check,
                     nfft_init_guru, nfft_trafo_direct, nfft_adjoint_direct,
                     nfft_trafo, nfft_adjoint, nfft_precompute_one_psi)
from cnfft3 cimport fftw_init_threads, fftw_cleanup, fftw_cleanup_threads
from cnfft3 cimport (PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI,
                     PRE_FULL_PSI, MALLOC_X, MALLOC_F_HAT, MALLOC_F,
                     FFT_OUT_OF_PLACE, FFTW_INIT, NFFT_SORT_NODES,
                     NFFT_OMP_BLOCKWISE_ADJOINT, PRE_ONE_PSI, FFTW_ESTIMATE,
                     FFTW_DESTROY_INPUT)

# Explicit aliases
ctypedef nfft_plan nfft_plan_double

# Generic function pointers defined for NFFT plans
ctypedef void *(*nfft_plan_generic_malloc)  (void *)
ctypedef void (*nfft_plan_generic_finalize) (void *)
ctypedef void (*nfft_plan_generic_init)     (void *, int, int *, int, int *,
                                             int, unsigned int, unsigned int)
ctypedef void (*nfft_plan_generic_trafo_direct)     (void *)
ctypedef void (*nfft_plan_generic_adjoint_direct)   (void *)
ctypedef void (*nfft_plan_generic_precompute)       (void *)


cdef inline void *nfft_plan_double_malloc(void *plan):
    return nfft_malloc(sizeof(nfft_plan_double))

cdef inline void nfft_plan_double_finalize():
    cdef nfft_plan_double *this_plan = <nfft_plan_double *>plan
    nfft_finalize(this_plan)
    
cdef inline void nfft_plan_double_init(void *plan, int d, int *N, int M,
                                       int *n, int m, unsigned int flags,
                                       unsigned int fftw_flags):
    nfft_init_guru(this_plan, d, N, M, n, m, flags, fftw_flags)

cdef inline void nfft_plan_double_(void *plan):
    cdef nfft_plan_double *this_plan = <nfft_plan_double *>plan
    nfft_trafo_direct(this_plan)

cdef inline void nfft_plan_double_adjoint_direct(void *plan):
    cdef nfft_plan_double *this_plan = <nfft_plan_double *>plan
    nfft_adjoint_direct(this_plan)

cdef inline void nfft_plan_double_precompute(void *plan):
    cdef nfft_plan_double *this_plan = <nfft_plan_double *>plan
    nfft_precompute_one_psi(this_plan)

cdef inline void nfft_plan_double_connect_arrays(void *plan, object f_hat,
                                                 object f, object x):
    cdef nfft_plan_double *this_plan = <nfft_plan_double *>plan
    this_plan.f_hat = <fftw_complex *>PyArray_DATA(f_hat)
    this_plan.f = <fftw_complex *>PyArray_DATA(f)
    this_plan.x = <double *>PyArray_DATA(x)

# Default values for flag parameters
cdef inline _default_flags():
    return PRE_PHI_HUT | PRE_PSI | FFTW_INIT | FFT_OUT_OF_PLACE

cdef inline _default_fftw_flags():
    return FFTW_ESTIMATE | FFTW_DESTROY_INPUT

# NFFT plan class
cdef class nfft_plan_proxy(mv_plan_proxy):
    cdef object _x
    cdef int _d
    cdef object _N
    cdef object _n
    cdef int _m
    cdef unsigned int _flags
    cdef unsigned int _fftw_flags
    cdef nfft_plan_generic_malloc           _plan_malloc
    cdef nfft_plan_generic_finalize         _plan_finalize
    cdef nfft_plan_generic_init             _plan_init
    cdef nfft_plan_generic_trafo_direct     _plan_trafo_direct
    cdef nfft_plan_generic_adjoint_direct   _plan_adjoint_direct
    cdef nfft_plan_generic_precompute       _plan_precompute
    cdef nfft_plan_generic_connect_arrays   _plan_connect_arrays
    cpdef init_1d(self, int, int)
    cpdef init_2d(self, int, int, int)
    cpdef init_3d(self, int, int, int, int)
    cpdef init(self, object, int)
    cpdef init_guru(self, object, int, object, int, unsigned int,
                    unsigned int)
    cpdef initialize_arrays(self)
    cpdef update_arrays(self, object f_hat, object f, object x)
    cpdef connect_arrays(self)
    cpdef trafo_direct(self)
    cpdef adjoint_direct(self)
    cpdef precompute(self)