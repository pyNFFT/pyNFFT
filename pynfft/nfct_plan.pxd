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
from cnfft3 cimport (nfct_plan, fftw_complex)
from cnfft3 cimport (nfct_malloc, nfct_finalize, nfct_free, nfct_check,
                     nfct_init_guru, nfct_trafo_direct, nfct_adjoint_direct,
                     nfct_trafo, nfct_adjoint, nfct_precompute_one_psi)
from cnfft3 cimport fftw_init_threads, fftw_cleanup, fftw_cleanup_threads
from cnfft3 cimport (PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI,
                     PRE_FULL_PSI, MALLOC_X, MALLOC_F_HAT, MALLOC_F,
                     FFT_OUT_OF_PLACE, FFTW_INIT, NFFT_SORT_NODES,
                     NFFT_OMP_BLOCKWISE_ADJOINT, PRE_ONE_PSI, FFTW_ESTIMATE,
                     FFTW_DESTROY_INPUT)

# Default values for flag parameters
cdef inline _default_nfct_flags():
    return PRE_PHI_HUT | PRE_PSI | FFTW_INIT | FFT_OUT_OF_PLACE

cdef inline _default_fftw_flags():
    return FFTW_ESTIMATE | FFTW_DESTROY_INPUT

# NFCT plan class
cdef class nfct_plan_proxy(mv_plan_proxy):
    cdef object _x
    cdef int _d
    cdef object _N
    cdef object _n
    cdef int _m
    cdef unsigned int _nfct_flags
    cdef unsigned int _fftw_flags
    cpdef init_1d(self, int N, int M)    
    cpdef init_2d(self, int N1, int N2, int M)
    cpdef init_3d(self, int N1, int N2, int N3, int M)
    cpdef init(self, object N, int M)
    cpdef init_guru(self, object N, int M, object n, int m,
                    unsigned int nfct_flags, unsigned int fftw_flags)
    cpdef initialize_arrays(self)
    cpdef update_arrays(self, object f_hat, object f, object x)
    cpdef connect_arrays(self)
    cpdef trafo_direct(self)
    cpdef adjoint_direct(self)
    cpdef precompute(self)
