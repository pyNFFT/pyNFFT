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
from cnfft3 cimport (PRE_PHI_HUT, PRE_PSI, FFTW_INIT, FFT_OUT_OF_PLACE,
                     FFTW_ESTIMATE, FFTW_DESTROY_INPUT)

# Function pointers specific to NFFT plans
ctypedef void *(*_nfft_plan_malloc_func) ()
ctypedef void (*_nfft_plan_finalize_func) (void *)
ctypedef void (*_nfft_plan_init_guru_func) (void *, int, int *, int, int *,
                                            int, unsigned int, unsigned int)
ctypedef void (*_nfft_plan_trafo_direct_func) (void *) nogil
ctypedef void (*_nfft_plan_adjoint_direct_func) (void *) nogil
ctypedef void (*_nfft_plan_precompute_one_psi_func) (void *) nogil
ctypedef const char *(*_nfft_plan_check_func) (void *)
ctypedef void (*_nfft_plan_connect_arrays_func) (void *, object, object, object)

# Default values for flag parameters
cdef inline _default_nfft_flags():
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
    cdef unsigned int _nfft_flags
    cdef unsigned int _fftw_flags

    cdef _nfft_plan_malloc_func _plan_malloc
    cdef _nfft_plan_finalize_func _plan_finalize
    cdef _nfft_plan_init_guru_func _plan_init_guru
    cdef _nfft_plan_trafo_direct_func _plan_trafo_direct
    cdef _nfft_plan_adjoint_direct_func _plan_adjoint_direct
    cdef _nfft_plan_precompute_one_psi_func _plan_precompute
    cdef _nfft_plan_check_func _plan_check    
    cdef _nfft_plan_connect_arrays_func _plan_connect_arrays  
    
    cpdef init_1d(self, int N, int M)    
    cpdef init_2d(self, int N1, int N2, int M)
    cpdef init_3d(self, int N1, int N2, int N3, int M)
    cpdef init(self, object N, int M)
    cpdef init_guru(self, object N, int M, object n, int m,
                    unsigned int nfft_flags, unsigned int fftw_flags)
    cpdef initialize_arrays(self)
    cpdef update_arrays(self, object f_hat, object f, object x)
    cpdef trafo_direct(self)
    cpdef adjoint_direct(self)
    cpdef precompute(self)
    cpdef check(self)