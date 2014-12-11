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

# public API
__all__ = ('nfft_plan_proxy', 'nfft_plan_flags', 'fftw_plan_flags')

from nfft_plan cimport *
from copy import copy

# Import numpy Python and C-API
import numpy
cimport numpy

### Module initialization
# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
numpy.import_array()

# Necessary for usage of threaded FFTW:
# - call init_threads on module import
fftw_init_threads()
# - define cleanup callback routine
cdef void _cleanup():
    fftw_cleanup()
    fftw_cleanup_threads()
# - register callback on module exit
cdef extern from *:
    int Py_AtExit(void (*callback)())
Py_AtExit(_cleanup)
###

# Dictionary mapping the NFFT plan flag names to their mask value
cdef dict nfft_plan_flags_dict = {
    'PRE_PHI_HUT'                   : PRE_PHI_HUT,
    'FG_PSI'                        : FG_PSI,
    'PRE_LIN_PSI'                   : PRE_LIN_PSI,
    'PRE_FG_PSI'                    : PRE_FG_PSI,
    'PRE_PSI'                       : PRE_PSI,
    'PRE_FULL_PSI'                  : PRE_FULL_PSI,
    'MALLOC_X'                      : MALLOC_X,
    'MALLOC_F_HAT'                  : MALLOC_F_HAT,
    'MALLOC_F'                      : MALLOC_F,
    'FFT_OUT_OF_PLACE'              : FFT_OUT_OF_PLACE,
    'FFTW_INIT'                     : FFTW_INIT,
    'NFFT_SORT_NODES'               : NFFT_SORT_NODES,
    'NFFT_OMP_BLOCKWISE_ADJOINT'    : NFFT_OMP_BLOCKWISE_ADJOINT,
    'PRE_ONE_PSI'                   : PRE_ONE_PSI,
    }
nfft_plan_flags = copy(nfft_plan_flags_dict)

# Dictionary mapping the FFTW plan flag names to their mask value
cdef dict fftw_plan_flags_dict = {
    'FFTW_ESTIMATE'                 : FFTW_ESTIMATE,
    'FFTW_DESTROY_INPUT'            : FFTW_DESTROY_INPUT,
    }
fftw_plan_flags = copy(fftw_plan_flags_dict)


cdef class nfft_plan_proxy(mv_plan_proxy):
    
    def __cinit__(self, dtype='cdouble', *args, **kwargs):
        idx = _nfft_plan_typenum_to_index[self._dtype.num]
        self._x                     = None
        self._plan_malloc           = _nfft_plan_malloc_func_list[idx]
        self._plan_finalize         = _nfft_plan_finalize_func_list[idx]
        self._plan_init_guru        = _nfft_plan_init_guru_func_list[idx]
        self._plan_trafo_direct     = _nfft_plan_trafo_direct_func_list[idx]
        self._plan_adjoint_direct   = _nfft_plan_adjoint_direct_func_list[idx]
        self._plan_precompute       = _nfft_plan_precompute_one_psi_func_list[idx]
        self._plan_check            = _nfft_plan_check_func_list[idx]
        self._plan_connect_arrays   = _nfft_plan_connect_arrays_func_list[idx]

    def __dealloc__(self):
        if self._is_initialized:
            self._plan_finalize(self._plan)

    cpdef init_1d(self, int N, int M):
        self.init((N,), M)
 
    cpdef init_2d(self, int N1, int N2, int M):
        self.init((N1, N2), M)

    cpdef init_3d(self, int N1, int N2, int N3, int M):
        self.init((N1, N2, N3), M)
   
    cpdef init(self, object N, int M):
        self.init_guru(N, M, [2 * Nt for Nt in N], 6, _default_nfft_flags(),
                       _default_fftw_flags())
    
    cpdef init_guru(self, object N, int M, object n, int m,
                    unsigned int nfft_flags, unsigned int fftw_flags):
        cdef int *N_ptr = NULL
        cdef int *n_ptr = NULL
        if not self._is_initialized:
            if len(N) != len(n):
                raise ValueError("incompatible geometry parameters")
            d = len(N)
            N_ptr = <int*>nfft_malloc(d*sizeof(int))
            for t, Nt in enumerate(N):
                N_ptr[t] = Nt
            n_ptr = <int*>nfft_malloc(d*sizeof(int))
            for t, nt in enumerate(n):
                n_ptr[t] = nt
            self._plan = self._plan_malloc()
            self._plan_init_guru(self._plan, d, N_ptr, M, n_ptr, m,
                                 nfft_flags, fftw_flags)
            nfft_free(N_ptr)
            nfft_free(n_ptr)
            self._N_total = numpy.prod(N) 
            self._M_total = M
            self._d = d
            self._N = N
            self._n = n
            self._m = m
            self._nfft_flags = nfft_flags            
            self._fftw_flags = fftw_flags
            self._is_initialized = True
            self.initialize_arrays()
        else:
            raise RuntimeError("plan is already initialized")

    cpdef initialize_arrays(self):
        if self._is_initialized:
            cplx_dtype = self.dtype
            real_dtype = complex_to_real_dtypes_table[cplx_dtype]
            self._f_hat = numpy.zeros(self.N, dtype=cplx_dtype)
            self._f = numpy.zeros(self.M_total, dtype=cplx_dtype)
            self._x = numpy.zeros([self.M_total, self.d], dtype=real_dtype)
            self._plan_connect_arrays(self._plan, self._f_hat, self._f, self._x)
        else:
            raise RuntimeError("plan is not initialized")

    cpdef update_arrays(self, object f_hat, object f, object x):
        if self._is_initialized:
            cplx_dtype = self.dtype
            real_dtype = complex_to_real_dtypes_table[cplx_dtype]
            if f_hat is not None:
                self._f_hat = numpy.ascontiguousarray(f_hat, dtype=cplx_dtype).reshape(self.N)
            if f is not None:
                self._f = numpy.ascontiguousarray(f, dtype=cplx_dtype).reshape(self.M_total)
            if x is not None:
                self._x = numpy.ascontiguousarray(x, dtype=real_dtype).reshape([self.M_total, self.d])
            self._plan_connect_arrays(self._plan, self._f_hat, self._f, self._x)
        else:
            raise RuntimeError("plan is not initialized")

    cpdef connect_arrays(self):
        pass

    cpdef trafo_direct(self):
        if self._is_initialized:    
            with nogil:
                self._plan_trafo_direct(self._plan)
        else:
            raise RuntimeError("plan is not initialized")
    
    cpdef adjoint_direct(self):
        if self._is_initialized:      
            with nogil:
                self._plan_adjoint_direct(self._plan)
        else:
            raise RuntimeError("plan is not initialized")
    
    cpdef precompute(self):
        if self._is_initialized:        
            with nogil:
                self._plan_precompute(self._plan)
        else:
            raise RuntimeError("plan is not initialized")

    cpdef check(self):
        cdef const char *c_errmsg
        cdef bytes py_errmsg
        if self._is_initialized:
            c_errmsg = self._plan_check(self._plan)
            if c_errmsg != NULL:
                py_errmsg = <bytes> c_errmsg
                raise RuntimeError(py_errmsg)
        else:
            raise RuntimeError("plan is not initialized")

    @property
    def d(self):
        return self._d

    @property
    def N(self):
        return self._N

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def nfft_flags(self):
        return self._nfft_flags

    @property
    def fftw_flags(self):
        return self._fftw_flags

    @property 
    def x(self):
        def __get__(self):
            return self._x
        def __set__(self, object value):
            if self._is_initialized:
                PyArray_CopyInto(self._x, value)