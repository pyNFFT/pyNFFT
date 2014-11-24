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
from cnfft3 cimport (PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI,
                     PRE_FULL_PSI, MALLOC_X, MALLOC_F_HAT, MALLOC_F,
                     FFT_OUT_OF_PLACE, FFTW_INIT, NFFT_SORT_NODES,
                     NFFT_OMP_BLOCKWISE_ADJOINT, PRE_ONE_PSI)
from cfftw3 cimport FFTW_ESTIMATE, FFTW_DESTROY_INPUT

# Function pointers to NFFT-specific plan management functions
ctypedef int    (*_plan_get_d_func) (void *)
ctypedef int   *(*_plan_get_N_func) (void *)
ctypedef int   *(*_plan_get_n_func) (void *)
ctypedef int    (*_plan_get_m_func) (void *)
ctypedef unsigned int (*_plan_get_nfft_flags_func) (void *)
ctypedef unsigned int (*_plan_get_fftw_flags_func) (void *)
ctypedef void  *(*_plan_get_x_func) (void *)
ctypedef void   (*_plan_set_x_func) (void *, void *)

ctypedef void   (*_plan_init_1d_func)   (void *, int, int)
ctypedef void   (*_plan_init_2d_func)   (void *, int, int, int)
ctypedef void   (*_plan_init_3d_func)   (void *, int, int, int, int)
ctypedef void   (*_plan_init_func)      (void *, int, int *, int)
ctypedef void   (*_plan_init_guru_func) (void *, int, int *, int, int*, int,
                                         unsigned int, unsigned int)
ctypedef void   (*_plan_trafo_direct_func)      (void *) nogil
ctypedef void   (*_plan_adjoint_direct_func)    (void *) nogil
ctypedef void   (*_plan_precompute_func)        (void *) nogil
ctypedef const char *(*_plan_check_func)        (void *)

# NFFT plan class
cdef class nfft_plan(mv_plan):
    cdef ndarray _x

    cdef _plan_get_d_func           _plan_get_d
    cdef _plan_get_N_func           _plan_get_N
    cdef _plan_get_n_func           _plan_get_n
    cdef _plan_get_m_func           _plan_get_m
    cdef _plan_get_nfft_flags_func  _plan_get_nfft_flags
    cdef _plan_get_fftw_flags_func  _plan_get_fftw_flags
    cdef _plan_get_x_func           _plan_get_x
    cdef _plan_set_x_func           _plan_set_x
    
    cdef _plan_init_1d_func         _plan_init_1d 
    cdef _plan_init_2d_func         _plan_init_2d
    cdef _plan_init_3d_func         _plan_init_3d 
    cdef _plan_init_func            _plan_init 
    cdef _plan_init_guru_func       _plan_init_guru
    cdef _plan_trafo_direct_func    _plan_trafo_direct
    cdef _plan_adjoint_direct_func  _plan_adjoint_direct
    cdef _plan_precompute_func      _plan_precompute 
    cdef _plan_check_func           _plan_check
    
    cpdef trafo_direct(self)
    cpdef adjoint_direct(self)
    cpdef precompute(self)
    cpdef precompute_one_psi(self)
    cpdef check(self)